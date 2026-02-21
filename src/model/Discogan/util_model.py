import sys;
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure
import src.utils.util_general as util_general
from src.utils.util_general import save_checkpoint
from sys import platform
from torch.cuda.amp import autocast, GradScaler
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as tm_ssim
from piq import vif_p as piq_vif
import lpips
import torch.nn.functional as F


def train_fn(
    G_AB, G_BA, D_A, D_B, loader, optimizer_G, optimizer_D_A, optimizer_D_B, adversarial_loss, pixelwise_loss,
    cycle_loss, g_scaler, d_scaler, device, cfg_trainer):

    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    loop = tqdm(loader, leave=True)

    running_G_loss = 0.0
    running_D_loss = 0.0
    n_samples = 0

    for idx, (real_A, real_B, _) in enumerate(loop):
        real_A = real_A.to(device, non_blocking=True)
        real_B = real_B.to(device, non_blocking=True)
        batch_size = real_A.size(0)
        n_samples += batch_size


        #  Train Generators
        optimizer_G.zero_grad(set_to_none=True)

        with autocast():
            # A -> B
            fake_B = G_AB(real_A)
            D_B_fake = D_B(fake_B)
            loss_GAN_AB = adversarial_loss(D_B_fake, torch.ones_like(D_B_fake))

            # B -> A
            fake_A = G_BA(real_B)
            D_A_fake = D_A(fake_A)
            loss_GAN_BA = adversarial_loss(D_A_fake, torch.ones_like(D_A_fake))

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2.0

            # Pixelwise translation loss (paired)
            loss_pixel = (
                pixelwise_loss(fake_A, real_A) +
                pixelwise_loss(fake_B, real_B)
            ) / 2.0

            # Cycle loss
            rec_A = G_BA(fake_B)  # A -> B -> A
            rec_B = G_AB(fake_A)  # B -> A -> B
            loss_cycle_A = cycle_loss(rec_A, real_A)
            loss_cycle_B = cycle_loss(rec_B, real_B)
            loss_cyc = (loss_cycle_A + loss_cycle_B) / 2.0

            # Loss totale generatori
            loss_G = loss_GAN + loss_pixel + loss_cyc

        g_scaler.scale(loss_G).backward()
        g_scaler.step(optimizer_G)
        g_scaler.update()


        #  Train Discriminators A + B
        optimizer_D_A.zero_grad(set_to_none=True)
        optimizer_D_B.zero_grad(set_to_none=True)

        with autocast():
            # D_A
            D_A_real = D_A(real_A)
            loss_real_A = adversarial_loss(D_A_real, torch.ones_like(D_A_real))


            D_A_fake = D_A(fake_A.detach())
            loss_fake_A = adversarial_loss(D_A_fake, torch.zeros_like(D_A_fake))

            loss_D_A = (loss_real_A + loss_fake_A) / 2.0

            # D_B
            D_B_real = D_B(real_B)
            loss_real_B = adversarial_loss(D_B_real, torch.ones_like(D_B_real))

            D_B_fake = D_B(fake_B.detach())
            loss_fake_B = adversarial_loss(D_B_fake, torch.zeros_like(D_B_fake))

            loss_D_B = (loss_real_B + loss_fake_B) / 2.0


            loss_D = 0.5 * (loss_D_A + loss_D_B)

        d_scaler.scale(loss_D).backward()
        d_scaler.step(optimizer_D_A)
        d_scaler.step(optimizer_D_B)
        d_scaler.update()

        running_G_loss += loss_G.item() * batch_size
        running_D_loss += loss_D.item() * batch_size

        loop.set_postfix(
            G_loss=running_G_loss / max(n_samples, 1),
            D_loss=running_D_loss / max(n_samples, 1),
        )

    avg_G_loss = running_G_loss / max(n_samples, 1)
    avg_D_loss = running_D_loss / max(n_samples, 1)

    return G_AB, G_BA, D_A, D_B, avg_G_loss, avg_D_loss


def eval_fn(gen_AB, loader, criterion, device, outputs_dir=None):

    gen_AB.eval()
    loop = tqdm(loader, leave=True)

    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for idx, (real_A, real_B, file_names) in enumerate(loop):
            real_A = real_A.to(device, non_blocking=True)
            real_B = real_B.to(device, non_blocking=True)
            batch_size = real_A.size(0)
            n_samples += batch_size

            with autocast():
                fake_B = gen_AB(real_A.float())

            if outputs_dir:
                os.makedirs(outputs_dir, exist_ok=True)
                for output, file_name in zip(fake_B, file_names):
                    filename_output = f"{file_name}_output.png"
                    save_image(output, os.path.join(outputs_dir, filename_output))

            loss = criterion(fake_B, real_B)
            running_loss += loss.item() * batch_size

    epoch_loss = running_loss / max(n_samples, 1)
    return epoch_loss


def train_discogan(
    G_AB, G_BA, D_A, D_B,
    data_loaders, early_stopping_criterion,
    optimizer_G, optimizer_D_A, optimizer_D_B,
    adversarial_loss, cycle_loss, pixelwise_loss,
    device, model_fold_dir, cfg_trainer
):

    g_scaler = GradScaler()
    d_scaler = GradScaler()

    best_G_AB_wts = copy.deepcopy(G_AB.state_dict())
    best_G_BA_wts = copy.deepcopy(G_BA.state_dict())
    best_D_A_wts = copy.deepcopy(D_A.state_dict())
    best_D_B_wts = copy.deepcopy(D_B.state_dict())

    best_loss = np.Inf
    best_epoch = 0

    history = {"train_loss": [], "val_loss": []}

    epochs_no_improve = 0
    early_stop = False
    numero_di_epoche = 0
    max_epochs = cfg_trainer["max_epochs"]
    warm_up = cfg_trainer["warm_up"]
    patience = cfg_trainer["early_stopping"]

    os.makedirs(model_fold_dir, exist_ok=True)

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        numero_di_epoche += 1

        # Train one epoch
        G_AB, G_BA, D_A, D_B, train_G_loss, train_D_loss = train_fn(
            G_AB=G_AB, G_BA=G_BA, D_A=D_A, D_B=D_B,
            loader=data_loaders['train'],
            optimizer_G=optimizer_G,
            optimizer_D_A=optimizer_D_A,
            optimizer_D_B=optimizer_D_B,
            adversarial_loss=adversarial_loss,
            pixelwise_loss=pixelwise_loss,
            cycle_loss=cycle_loss,
            g_scaler=g_scaler,
            d_scaler=d_scaler,
            device=device,
            cfg_trainer=cfg_trainer,
        )

        # Validation (A -> B)
        val_loss = eval_fn(
            gen_AB=G_AB,
            loader=data_loaders["val"],
            criterion=early_stopping_criterion,
            device=device,
            outputs_dir=None,
        )


        train_loss = eval_fn(
            gen_AB=G_AB,
            loader=data_loaders["train"],
            criterion=early_stopping_criterion,
            device=device,
            outputs_dir=None,
        )

        history["val_loss"].append(val_loss)
        history["train_loss"].append(train_loss)

        print(
            f"Epoch {epoch+1}: "
            f"train_metric={train_loss:.4f}, "
            f"val_metric={val_loss:.4f}, "
            f"G_loss={train_G_loss:.4f}, "
            f"D_loss={train_D_loss:.4f}"
        )

        # Early stopping after warm-up
        if epoch > warm_up:
            if val_loss < best_loss:
                best_epoch = epoch
                best_loss = val_loss
                best_G_AB_wts = copy.deepcopy(G_AB.state_dict())
                best_G_BA_wts = copy.deepcopy(G_BA.state_dict())
                best_D_A_wts = copy.deepcopy(D_A.state_dict())
                best_D_B_wts = copy.deepcopy(D_B.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'\nEarly Stopping! Total epochs: {epoch+1}')
                    early_stop = True

        if early_stop:
            break
    if numero_di_epoche % 1 == 0:
        print('numero di epoche finite: ' + str(numero_di_epoche))

    print('Training complete')
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Loss: {:4f}'.format(best_loss))


    G_AB.load_state_dict(best_G_AB_wts)
    G_BA.load_state_dict(best_G_BA_wts)
    D_A.load_state_dict(best_D_A_wts)
    D_B.load_state_dict(best_D_B_wts)

    # Save model
    save_checkpoint(G_AB, optimizer_G,
                    filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN_AB"]))
    save_checkpoint(G_BA, optimizer_G,
                    filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN_BA"]))
    save_checkpoint(D_A, optimizer_D_A,
                    filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DISC_A"]))
    save_checkpoint(D_B, optimizer_D_B,
                    filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DISC_B"]))

    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return G_AB, G_BA, D_A, D_B, history


def plot_training(history: pd.DataFrame, model_name: str, plot_training_dir: str):
    util_general.create_dir(os.path.join(plot_training_dir, model_name))
    out_dir = os.path.join(plot_training_dir, model_name)
    plt.figure(figsize=(8, 6))
    plt.plot(history["train_loss"], label="Training L1 (MRI→CT)")
    plt.plot(history["val_loss"],   label="Validation L1 (MRI→CT)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss (UNIT)")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(out_dir, "Loss.png"))
    plt.close()


def test_4metrics(G_AB, loader_test, mse_metric, psnr_metric, device, outputs_dir,
                  save_output):
    G_AB.eval()
    loop_test = tqdm(loader_test, leave=True)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_loss4 = 0.0

    with torch.no_grad():
        for idx, (lowenergy, recombined, file_names) in enumerate(loop_test):
            real_a = lowenergy.to(device)      # input
            real_b = recombined.to(device)     # target

            with torch.cuda.amp.autocast():
                fake_b = G_AB(real_a.float())

                if save_output:
                    for output, file_name in zip(fake_b, file_names):  # SAVE OUTPUTS PNG
                        filename_output = f"{file_name}_output.png"
                        save_image(output, os.path.join(outputs_dir, filename_output))


            fake_eval = fake_b.float()
            real_eval = real_b.float()


            if fake_eval.min() < 0 or fake_eval.max() > 1:
                fake_eval = (fake_eval + 1) / 2
            if real_eval.min() < 0 or real_eval.max() > 1:
                real_eval = (real_eval + 1) / 2

            fake_eval = fake_eval.clamp(0, 1)
            real_eval = real_eval.clamp(0, 1)

            # ---- METRICS ----
            loss1 = mse_metric(fake_eval, real_eval)
            loss2 = psnr_metric(fake_eval, real_eval)  # se usi piq.psnr, passa anche data_range=1.0 nella definizione
            loss3 = vif_p(fake_eval, real_eval, data_range=1.0)
            loss4 = structural_similarity_index_measure(fake_eval, real_eval, data_range=1.0)

            # statistics
            batch_size = lowenergy.size(0)
            running_loss1 += loss1.item() * batch_size
            running_loss2 += loss2.item() * batch_size
            running_loss3 += loss3.item() * batch_size
            running_loss4 += loss4.item() * batch_size

    num_samples = len(loader_test.dataset) if hasattr(loader_test, "dataset") else len(loader_test)
    epoch_loss1 = running_loss1 / num_samples
    epoch_loss2 = running_loss2 / num_samples
    epoch_loss3 = running_loss3 / num_samples
    epoch_loss4 = running_loss4 / num_samples
    return epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4
@torch.no_grad()
def test_discogan(G_AB, loader_test, mse_metric, psnr_metric, device,
                  outputs_dir=None, save_outputs=False, *,
                  amp=True, lpips_backbone="alex", is_splits=10,
                  return_per_image=False):

    G_AB.eval()

    metric_sums = {"mse":0.0,"psnr":0.0,"vif":0.0,"ssim":0.0,"lpips":0.0}
    per_image = {"mse":[], "psnr":[], "vif":[], "ssim":[], "lpips":[]}
    n_images = 0

    # LPIPS
    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    # Inception (IS/FID)
    inc_cls  = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat.fc = nn.Identity()

    logits_all, feats_real_all, feats_fake_all = [], [], []

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type=="cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for _, batch in enumerate(pbar):
        x1, x2, file_names = batch
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)
        bsz = x1.size(0)
        n_images += bsz

        # forward (generator)
        with scaler_ctx():
            fake_x2 = G_AB(x1)


        fake_01 = (fake_x2 + 1.0) / 2.0
        real_01 = (x2       + 1.0) / 2.0

        fake_01 = fake_01.clamp(0.0, 1.0)
        real_01 = real_01.clamp(0.0, 1.0)


        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_01, file_names):
                save_image(out_img, os.path.join(outputs_dir, f"{name}_output.png"))


        mse_val  = _to_scalar(mse_metric(fake_01, real_01))
        psnr_val = _to_scalar(psnr_metric(fake_01, real_01))

        with torch.cuda.amp.autocast(enabled=False):
            vif_val  = _to_scalar(piq_vif(fake_01.float(), real_01.float(), data_range=1.0))
            ssim_val = _to_scalar(tm_ssim(fake_01, real_01, data_range=1.0))


        a_lp = _prep_for_lpips(fake_01)
        b_lp = _prep_for_lpips(real_01)
        lpips_val = _to_scalar(lpips_fn(a_lp, b_lp).mean())


        metric_sums["mse"]   += mse_val * bsz
        metric_sums["psnr"]  += psnr_val * bsz
        metric_sums["vif"]   += vif_val * bsz
        metric_sums["ssim"]  += ssim_val * bsz
        metric_sums["lpips"] += lpips_val * bsz

        if return_per_image:
            per_image["mse"].append(mse_val)
            per_image["psnr"].append(psnr_val)
            per_image["vif"].append(vif_val)
            per_image["ssim"].append(ssim_val)
            per_image["lpips"].append(lpips_val)

        # --------------------------------------------------------
        # Inception features for IS / FID
        # --------------------------------------------------------
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fake_inc = _prep_for_inception(fake_01).float().to(device)
                real_inc = _prep_for_inception(real_01).float().to(device)

                logits = inc_cls(fake_inc)
                logits_all.append(logits.detach().cpu())

                f_fake = inc_feat(fake_inc)
                f_real = inc_feat(real_inc)
                feats_fake_all.append(f_fake.detach().cpu().numpy())
                feats_real_all.append(f_real.detach().cpu().numpy())


    results = {}
    denom = max(n_images, 1)
    for k, v in metric_sums.items():
        results[k] = v / denom

    # FID
    if len(feats_fake_all) > 0 and len(feats_real_all) > 0:
        Ffake = np.concatenate(feats_fake_all, axis=0)
        Freal = np.concatenate(feats_real_all, axis=0)
        mu_f, sig_f = Ffake.mean(axis=0), np.cov(Ffake, rowvar=False)
        mu_r, sig_r = Freal.mean(axis=0), np.cov(Freal, rowvar=False)
        results["fid"] = _frechet_distance(mu_r, sig_r, mu_f, sig_f)
    else:
        results["fid"] = float("nan")


    if len(logits_all) > 0:
        logits_cat = torch.cat(logits_all, dim=0)
        is_mean, is_std = _inception_score_from_logits(logits_cat, splits=is_splits)
        results["is_mean"] = float(is_mean)
        results["is_std"]  = float(is_std)
    else:
        results["is_mean"] = float("nan")
        results["is_std"]  = float("nan")

    if return_per_image:
        results["per_image"] = per_image

    return results



# ---------------- helpers ----------------

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.numel()==1: return float(x.item())
        return float(x.mean().item())
    return float(x)

def _ensure_3ch(x):
    if x.size(1)==3: return x
    return x.repeat(1,3,1,1)

def _prep_for_lpips(x):
    x = x.clamp(0,1)
    x = _ensure_3ch(x)
    return x*2.0 - 1.0

def _prep_for_inception(x):
    x = _ensure_3ch(x.clamp(0,1))
    x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485,0.456,0.406])[None,:,None,None]
    std  = x.new_tensor([0.229,0.224,0.225])[None,:,None,None]
    return (x-mean)/std

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0])*eps
        covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0*tr_covmean
    return float(fid)

def _inception_score_from_logits(logits, splits=10):
    N = logits.size(0)
    splits = max(1, min(splits, N))
    probs = torch.softmax(logits, dim=1)
    split_scores = []
    for k in range(splits):
        part = probs[k*(N//splits):(k+1)*(N//splits)]
        py = part.mean(dim=0, keepdim=True)
        kl = (part*(part.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(dim=1)
        split_scores.append(torch.exp(kl.mean()).item())
    return float(np.mean(split_scores)), float(np.std(split_scores))



