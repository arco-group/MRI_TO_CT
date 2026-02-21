import sys;
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
from tqdm import tqdm
from torchvision.utils import save_image
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure
from sys import platform
import sys; print('Python %s on %s' % (sys.version, sys.platform))
import copy
import pandas as pd
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
import src.utils.util_general as util_general
from src.utils.util_general import save_checkpoint
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as tm_ssim
from piq import multi_scale_ssim as piq_ms_ssim, vif_p as piq_vif
import lpips
import torch.nn.functional as F


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, device, cfg_trainer):
    gen.train()
    disc.train()

    loop = tqdm(loader, leave=True)

    for idx, (x, y, id) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            #mri e ct
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

            L1 = l1_loss(y_fake, y) * cfg_trainer["L1_LAMBDA"]

            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    return disc, gen


def eval_fn(gen, loader, criterion, device, outputs_dir):
    gen.eval()

    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    for idx, (x, y, file_names) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast():
            y_fake = gen(x)

            if outputs_dir:
                for output, file_name in zip(y_fake, file_names):
                    # SAVE OUTPUTS PNG
                    filename_output = "%s_output.png" % (file_name)
                    save_image(output, os.path.join(outputs_dir, filename_output))

        loss = criterion(y_fake, y)
        # statistics
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(loader)
    return epoch_loss


def train_pix2pix(gen, disc, data_loaders, early_stopping_criterion, opt_disc, opt_gen, L1, bce, model_fold_dir,
                  cfg_trainer, device):
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best_gen_wts = copy.deepcopy(gen.state_dict())
    best_disc_wts = copy.deepcopy(disc.state_dict())
    best_loss = np.Inf


    history = {'train_loss' : [],'val_loss': []}

    epochs_no_improve = 0
    early_stop = False
    numero_di_epoche = 0
    for epoch in range(cfg_trainer["max_epochs"]):
        # Train one epoch
        numero_di_epoche += 1

        disc, gen = train_fn(disc=disc, gen=gen, loader=data_loaders['train'], opt_disc=opt_disc, opt_gen=opt_gen,
                             l1_loss=L1,
                             bce=bce, g_scaler=g_scaler, d_scaler=d_scaler, device=device, cfg_trainer=cfg_trainer)

                             # Val
        epoch_loss = eval_fn(gen=gen, loader=data_loaders['val'], criterion=early_stopping_criterion, device=device,
                             outputs_dir=False)
        epoch_train_loss = eval_fn(gen=gen, loader=data_loaders['train'], criterion=early_stopping_criterion, device=device,
                             outputs_dir=False)

        history['val_loss'].append(epoch_loss)
        history['train_loss'].append(epoch_train_loss)#stiamo riempiendo le chiavi di History
        if epoch > cfg_trainer['warm_up']:
            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_gen_wts = copy.deepcopy(gen.state_dict())
                best_disc_wts = copy.deepcopy(disc.state_dict())
                epochs_no_improve = 0


            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= cfg_trainer["early_stopping"]:
                    print(f'\nEarly Stopping! Total epochs: {epoch}%')
                    early_stop = True
            if early_stop:
                break
        if numero_di_epoche%1 == 0:
            print('numero di epoche finite: '+str(numero_di_epoche))

    print('Training e validation complete')
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best epoch: {:0f}'.format(best_epoch))


    gen.load_state_dict(best_gen_wts)
    disc.load_state_dict(best_disc_wts)

    # Save model
    save_checkpoint(gen, opt_gen, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN"]))
    save_checkpoint(disc, opt_disc, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DISC"]))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return gen, disc, history


def plot_training(history, model_name, plot_training_dir):

    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)


    plt.figure(figsize=(8, 6))


    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')


    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)


    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


def test_5metrics(gen, loader, mse_metric, psnr_metric, device, outputs_dir,
                  save_output, fid_metric):
    gen.eval()
    loop = tqdm(loader, leave=True)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_loss4 = 0.0

    fid_metric.reset()

    with torch.no_grad():
        for idx, (x, y, file_names) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            with torch.cuda.amp.autocast():
                y_fake = gen(x)

                if save_output:
                    for output, file_name in zip(y_fake, file_names):
                        filename_output = f"{file_name}_output.png"
                        save_image(output, os.path.join(outputs_dir, filename_output))

            # metrics
            loss1 = mse_metric(y_fake, y)
            loss2 = psnr_metric(y_fake, y)

            if platform.system() == 'Windows':
                loss3 = vif_p(y_fake, y)
                loss4 = structural_similarity_index_measure(y_fake, y)
            else:
                loss3 = vif_p(y_fake.float(), y.float())
                loss4 = structural_similarity_index_measure(
                    y_fake.to(torch.float16), y.to(torch.float16)
                )

            # --- FID ---

            fake_for_fid = (y_fake.clamp(0, 1) * 255).to(torch.uint8)
            real_for_fid = (y.clamp(0, 1) * 255).to(torch.uint8)


            fake_for_fid = fake_for_fid.repeat(1, 3, 1, 1)
            real_for_fid = real_for_fid.repeat(1, 3, 1, 1)

            # update FID
            fid_metric.update(real_for_fid, real=True)
            fid_metric.update(fake_for_fid, real=False)

            # number of epoch
            running_loss1 += loss1.item() * x.size(0)
            running_loss2 += loss2.item() * x.size(0)
            running_loss3 += loss3.item() * x.size(0)
            running_loss4 += loss4.item() * x.size(0)

    # compute epoch metrics
    n_samples = len(loader.dataset)
    epoch_loss1 = running_loss1 / n_samples
    epoch_loss2 = running_loss2 / n_samples
    epoch_loss3 = running_loss3 / n_samples
    epoch_loss4 = running_loss4 / n_samples

    epoch_fid = fid_metric.compute().item()

    return epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_fid

def test_4metrics(gen, loader, mse_metric, psnr_metric, device, outputs_dir,
                  save_output):
    gen.eval()
    loop = tqdm(loader, leave=True)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_loss4 = 0.0

    with torch.no_grad():
        for idx, (x, y, file_names) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                if save_output:
                    for output, file_name in zip(y_fake, file_names):  # SAVE OUTPUTS PNG
                        filename_output = "%s_output.png" % (file_name)
                        save_image(output, os.path.join(outputs_dir, filename_output))


            loss1 = mse_metric(y_fake, y)
            loss2 = psnr_metric(y_fake, y)
            if platform == 'win32':  # windows
                loss3 = vif_p(y_fake, y)
                loss4 = structural_similarity_index_measure(y_fake, y)
            else: # alvis
                loss3 = vif_p(y_fake.float(), y.float())
                loss4 = structural_similarity_index_measure(y_fake.to(torch.float16), y.to(torch.float16))

            # statistics
            running_loss1 += loss1.item() * x.size(0)
            running_loss2 += loss2.item() * x.size(0)
            running_loss3 += loss3.item() * x.size(0)
            running_loss4 += loss4.item() * x.size(0)

    epoch_loss1 = running_loss1 / len(loader)
    epoch_loss2 = running_loss2 / len(loader)
    epoch_loss3 = running_loss3 / len(loader)
    epoch_loss4 = running_loss4 / len(loader)

    return epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4


def test_pix2pix(gen, loader_test, mse_metric, psnr_metric, device, outputs_dir=None,
               save_outputs=False, *, amp=True, lpips_backbone="alex",
               is_splits=10, return_per_image=False):

    gen.eval()

    # --- accumulators (somme pesate per numero di immagini) ---
    metric_sums = {
        "mse": 0.0,
        "psnr": 0.0,
        "vif": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "ms_ssim": 0.0,
    }

    # --- NEW: accumulo per immagine ---
    per_image = {
        "mse": [],
        "psnr": [],
        "vif": [],
        "ssim": [],
        "lpips": []
    }

    n_images = 0

    # --- metric models ---
    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    # Inception per IS / FID
    inc_cls = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat.fc = nn.Identity()  # features 2048

    logits_all = []     # IS (logits su fake)
    feats_real_all = [] # FID (features 2048 su real)
    feats_fake_all = [] # FID (features 2048 su fake)

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for batch_idx, batch in enumerate(pbar):

        x1, x2, file_names = batch
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)
        bsz = x1.size(0)
        n_images += bsz

        # --- forward generator ---
        with scaler_ctx():
            fake_x2 = gen(x1)


        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_x2, file_names):
                save_image(
                    out_img.clamp(0, 1),
                    os.path.join(outputs_dir, f"{name}_output.png"),
                )


        mse_val  = _to_scalar(mse_metric(fake_x2, x2))
        psnr_val = _to_scalar(psnr_metric(fake_x2, x2))

        # VIF (piq)
        with torch.cuda.amp.autocast(enabled=False):
            vif_val = _to_scalar(piq_vif(fake_x2.float(), x2.float()))

        # SSIM (torchmetrics)
        ssim_val = _to_scalar(tm_ssim(fake_x2, x2, data_range=1.0))

        # LPIPS
        a_lp = _prep_for_lpips(fake_x2)
        b_lp = _prep_for_lpips(x2)
        lpips_val = _to_scalar(lpips_fn(a_lp, b_lp).mean())

        # MS-SSIM (piq)
        a_3c = _ensure_3ch(fake_x2)
        b_3c = _ensure_3ch(x2)
        with torch.cuda.amp.autocast(enabled=False):
            ms_ssim_val = _to_scalar(
                piq_ms_ssim(a_3c.float(), b_3c.float(), data_range=1.0, reduction="mean")
            )


        metric_sums["mse"]     += mse_val * bsz
        metric_sums["psnr"]    += psnr_val * bsz
        metric_sums["vif"]     += vif_val * bsz
        metric_sums["ssim"]    += ssim_val * bsz
        metric_sums["lpips"]   += lpips_val * bsz
        metric_sums["ms_ssim"] += ms_ssim_val * bsz


        if return_per_image:
            per_image["mse"].append(mse_val)
            per_image["psnr"].append(psnr_val)
            per_image["vif"].append(vif_val)
            per_image["ssim"].append(ssim_val)
            per_image["lpips"].append(lpips_val)


        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fake_inc = _prep_for_inception(fake_x2).float().to(device)
                real_inc = _prep_for_inception(x2).float().to(device)


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

    # --- FID ---
    if len(feats_fake_all) > 0 and len(feats_real_all) > 0:
        Ffake = np.concatenate(feats_fake_all, axis=0)
        Freal = np.concatenate(feats_real_all, axis=0)
        mu_f, sig_f = Ffake.mean(axis=0), np.cov(Ffake, rowvar=False)
        mu_r, sig_r = Freal.mean(axis=0), np.cov(Freal, rowvar=False)
        results["fid"] = _frechet_distance(mu_r, sig_r, mu_f, sig_f)
    else:
        results["fid"] = float("nan")

    # --- Inception Score ---
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



# ----------------- helper -----------------

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.numel() == 1:
            return float(x.item())
        return float(x.mean().item())
    return float(x)

def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W] with C in {1,3}
    if x.size(1) == 3:
        return x
    return x.repeat(1, 3, 1, 1)

def _prep_for_lpips(x: torch.Tensor) -> torch.Tensor:

    x = x.clamp(0, 1)
    x = _ensure_3ch(x)
    x = x * 2.0 - 1.0
    return x

def _prep_for_inception(x: torch.Tensor) -> torch.Tensor:

    x = _ensure_3ch(x.clamp(0, 1))
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std  = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (x - mean) / std
    return x

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID: d^2 = ||mu1-mu2||^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma1*Sigma2)^{1/2})
    """
    from scipy import linalg
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)

def _inception_score_from_logits(logits: torch.Tensor, splits: int = 10):

    N = logits.size(0)
    splits = max(1, min(splits, N))
    probs = torch.softmax(logits, dim=1)  # [N,1000]
    split_scores = []
    for k in range(splits):
        part = probs[k * (N // splits):(k + 1) * (N // splits), :]
        py = part.mean(dim=0, keepdim=True)  # p(y)
        # KL(p(y|x) || p(y))
        kl = (part * (part.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(dim=1)
        split_scores.append(torch.exp(kl.mean()).item())
    return float(np.mean(split_scores)), float(np.std(split_scores))


