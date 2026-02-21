import sys
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
from torchmetrics import PeakSignalNoiseRatio

import src.utils.util_general as util_general
from src.utils.util_general import save_checkpoint
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

from piq import vif_p as piq_vif
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as tm_ssim
from torchvision.models.inception import inception_v3, Inception_V3_Weights
import lpips

def train_fn(
    Enc1, Dec1, Enc2, Dec2, D1, D2,
    loader,
    opt_G, opt_D1, opt_D2,
    l1_recon,
    lambdas,
    device,
    scaler_G: torch.cuda.amp.GradScaler,
    scaler_D1: torch.cuda.amp.GradScaler,
    scaler_D2: torch.cuda.amp.GradScaler
):

    Enc1.train()
    Dec1.train()
    Enc2.train()
    Dec2.train()
    D1.train()
    D2.train()

    lambda_gan   = lambdas["LAMBDA_GAN"]
    lambda_id    = lambdas["LAMBDA_ID"]
    lambda_style = lambdas["LAMBDA_STYLE"]
    lambda_cont  = lambdas["LAMBDA_CONTENT"]
    lambda_cyc   = lambdas["LAMBDA_CYCLE"]

    loop = tqdm(loader, leave=True)

    running_loss_G = 0.0
    running_loss_D = 0.0

    for idx, (mri, ct, _) in enumerate(loop):
        X1 = mri.to(device)
        X2 = ct.to(device)


        style_dim = lambdas["STYLE_DIM"]
        style_1 = torch.randn(X1.size(0), style_dim, 1, 1, device=device)
        style_2 = torch.randn(X2.size(0), style_dim, 1, 1, device=device)

        # -------------------------------
        #  Train Encoders & Decoders
        # -------------------------------
        opt_G.zero_grad()

        with torch.cuda.amp.autocast():
            # encoding
            c1, s1 = Enc1(X1)  # MRI
            c2, s2 = Enc2(X2)  # CT


            X11 = Dec1(c1, s1)
            X22 = Dec2(c2, s2)

            # cross-translation
            X21 = Dec1(c2, style_1)  # CT -> MRI
            X12 = Dec2(c1, style_2)  # MRI -> CT


            c21, s21 = Enc1(X21)
            c12, s12 = Enc2(X12)
            X121 = Dec1(c12, s1) if lambda_cyc > 0 else 0
            X212 = Dec2(c21, s2) if lambda_cyc > 0 else 0

            # Loss GAN
            loss_GAN_1 = lambda_gan * D1.compute_loss(X21, 1.0)
            loss_GAN_2 = lambda_gan * D2.compute_loss(X12, 1.0)

            # Identity
            loss_ID_1 = lambda_id * l1_recon(X11, X1)
            loss_ID_2 = lambda_id * l1_recon(X22, X2)

            # style
            loss_s_1 = lambda_style * l1_recon(s21, style_1)
            loss_s_2 = lambda_style * l1_recon(s12, style_2)

            # content
            loss_c_1 = lambda_cont * l1_recon(c12, c1.detach())
            loss_c_2 = lambda_cont * l1_recon(c21, c2.detach())


            if lambda_cyc > 0:
                loss_cyc_1 = lambda_cyc * l1_recon(X121, X1)
                loss_cyc_2 = lambda_cyc * l1_recon(X212, X2)
            else:
                loss_cyc_1 = 0
                loss_cyc_2 = 0

            loss_G = (
                loss_GAN_1 + loss_GAN_2
                + loss_ID_1 + loss_ID_2
                + loss_s_1 + loss_s_2
                + loss_c_1 + loss_c_2
                + loss_cyc_1 + loss_cyc_2
            )

        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------
        opt_D1.zero_grad()
        with torch.cuda.amp.autocast():
            loss_D1 = D1.compute_loss(X1, 1.0) + D1.compute_loss(X21.detach(), 0.0)
        scaler_D1.scale(loss_D1).backward()
        scaler_D1.step(opt_D1)
        scaler_D1.update()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------
        opt_D2.zero_grad()
        with torch.cuda.amp.autocast():
            loss_D2 = D2.compute_loss(X2, 1.0) + D2.compute_loss(X12.detach(), 0.0)
        scaler_D2.scale(loss_D2).backward()
        scaler_D2.step(opt_D2)
        scaler_D2.update()

        loss_D = loss_D1 + loss_D2

        running_loss_G += loss_G.item() * X1.size(0)
        running_loss_D += loss_D.item() * X1.size(0)

        loop.set_postfix(G=loss_G.item(), D=loss_D.item())

    epoch_loss_G = running_loss_G / len(loader.dataset)
    epoch_loss_D = running_loss_D / len(loader.dataset)
    return Enc1, Dec1, Enc2, Dec2, D1, D2, epoch_loss_G, epoch_loss_D


def eval_fn(Enc1, Dec2, loader_val, l1_recon, device, style_dim: int):

    Enc1.eval()
    Dec2.eval()
    running_loss = 0.0
    loop = tqdm(loader_val, leave=True)
    with torch.no_grad():
        for mri, ct, _ in loop:
            X1 = mri.to(device)
            X2 = ct.to(device)
            style_2 = torch.randn(X2.size(0), style_dim, 1, 1, device=device)
            with torch.cuda.amp.autocast():
                c1, _ = Enc1(X1)
                X12 = Dec2(c1, style_2)  # MRI->CT
                loss = l1_recon(X12, X2)
            running_loss += loss.item() * X1.size(0)
    epoch_loss = running_loss / len(loader_val.dataset)
    return epoch_loss


def train_munit(
    Enc1, Dec1, Enc2, Dec2, D1, D2,
    data_loaders,
    opt_G, opt_D1, opt_D2,
    l1_recon,
    lambdas,
    model_fold_dir,
    cfg_trainer,
    device,
):

    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D1 = torch.cuda.amp.GradScaler()
    scaler_D2 = torch.cuda.amp.GradScaler()

    best_Enc1_w = copy.deepcopy(Enc1.state_dict())
    best_Dec1_w = copy.deepcopy(Dec1.state_dict())
    best_Enc2_w = copy.deepcopy(Enc2.state_dict())
    best_Dec2_w = copy.deepcopy(Dec2.state_dict())
    best_D1_w   = copy.deepcopy(D1.state_dict())
    best_D2_w   = copy.deepcopy(D2.state_dict())

    best_loss = np.Inf
    best_epoch = 0
    history = {'train_loss_G': [], 'train_loss_D': [], 'val_loss': []}
    epochs_no_improve = 0
    early_stop = False
    numero_di_epoche = 0

    for epoch in range(cfg_trainer["max_epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg_trainer['max_epochs']}")
        numero_di_epoche += 1

        Enc1, Dec1, Enc2, Dec2, D1, D2, train_G, train_D = train_fn(
            Enc1, Dec1, Enc2, Dec2, D1, D2,
            loader=data_loaders['train'],
            opt_G=opt_G, opt_D1=opt_D1, opt_D2=opt_D2,
            l1_recon=l1_recon,
            lambdas=lambdas,
            device=device,
            scaler_G=scaler_G,
            scaler_D1=scaler_D1,
            scaler_D2=scaler_D2,
        )

        val_loss = eval_fn(
            Enc1=Enc1, Dec2=Dec2,
            loader_val=data_loaders['val'],
            l1_recon=l1_recon,
            device=device,
            style_dim=lambdas["STYLE_DIM"],
        )

        history['train_loss_G'].append(train_G)
        history['train_loss_D'].append(train_D)
        history['val_loss'].append(val_loss)

        print(f"Train G: {train_G:.4f}  Train D: {train_D:.4f}  Val L1(MRI->CT): {val_loss:.4f}")

        if epoch > cfg_trainer["warm_up"]:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_Enc1_w = copy.deepcopy(Enc1.state_dict())
                best_Dec1_w = copy.deepcopy(Dec1.state_dict())
                best_Enc2_w = copy.deepcopy(Enc2.state_dict())
                best_Dec2_w = copy.deepcopy(Dec2.state_dict())
                best_D1_w   = copy.deepcopy(D1.state_dict())
                best_D2_w   = copy.deepcopy(D2.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg_trainer["early_stopping"]:
                    print(f"\nEarly Stopping! Epoch: {epoch}")
                    early_stop = True
        if early_stop:
            break
    if numero_di_epoche % 1 == 0:
        print('number of total epochs: ' + str(numero_di_epoche))

    print('Training complete')
    print(f'Best epoch: {best_epoch}  Best val loss: {best_loss:.4f}')

    Enc1.load_state_dict(best_Enc1_w)
    Dec1.load_state_dict(best_Dec1_w)
    Enc2.load_state_dict(best_Enc2_w)
    Dec2.load_state_dict(best_Dec2_w)
    D1.load_state_dict(best_D1_w)
    D2.load_state_dict(best_D2_w)

    # Save best model
    save_checkpoint(Enc1, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_ENC1"]))
    save_checkpoint(Dec1, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DEC1"]))
    save_checkpoint(Enc2, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_ENC2"]))
    save_checkpoint(Dec2, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DEC2"]))
    save_checkpoint(D1, opt_D1, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D1"]))
    save_checkpoint(D2, opt_D2, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D2"]))

    history = pd.DataFrame.from_dict(history, orient='index').transpose()
    return Enc1, Dec1, Enc2, Dec2, D1, D2, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss_G'], label='Train G')
    plt.plot(history['train_loss_D'], label='Train D')
    plt.plot(history['val_loss'], label='Val L1 (MRI->CT)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


def test_4metrics(
    Enc1, Dec2,
    loader_test,
    device,
    outputs_dir,
    save_output,
):

    Enc1.eval()
    Dec2.eval()

    mse_metric  = torch.nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    print("test set MUNIT:", len(loader_test.dataset))

    loop_test = tqdm(loader_test, leave=True)
    running_mse = running_psnr = running_vif = running_ssim = 0.0

    with torch.no_grad():
        for mri, ct, file_names in loop_test:
            X1 = mri.to(device)   # MRI
            X2 = ct.to(device)


            style_2 = torch.randn(X1.size(0), 8, 1, 1, device=device)

            with torch.cuda.amp.autocast():
                c1, _ = Enc1(X1)
                fake_ct = Dec2(c1, style_2)


            if save_output:
                fake_ct_png = (fake_ct.float() + 1.0) / 2.0
                fake_ct_png = torch.clamp(fake_ct_png, 0.0, 1.0)
                for out_img, fname in zip(fake_ct_png, file_names):
                    filename_output = f"{fname}_munit_ct.png"
                    save_image(out_img, os.path.join(outputs_dir, filename_output))


            fake_ct_01 = (fake_ct.float() + 1.0) / 2.0
            X2_01      = (X2.float()      + 1.0) / 2.0


            fake_ct_01 = torch.clamp(fake_ct_01, 0.0, 1.0)
            X2_01      = torch.clamp(X2_01,      0.0, 1.0)


            mse  = mse_metric(fake_ct_01, X2_01)
            psnr = psnr_metric(fake_ct_01, X2_01)
            vif  = vif_p(fake_ct_01, X2_01, data_range=1.0)
            ssim = structural_similarity_index_measure(
                fake_ct_01, X2_01, data_range=1.0
            )

            running_mse   += mse.item()  * X1.size(0)
            running_psnr  += psnr.item() * X1.size(0)
            running_vif   += vif.item()  * X1.size(0)
            running_ssim  += ssim.item() * X1.size(0)

    n = len(loader_test.dataset)
    return (
        running_mse  / n,
        running_psnr / n,
        running_vif  / n,
        running_ssim / n,
    )
def test_munit(Enc1, Dec2, loader_test, mse_metric, psnr_metric, device,
               outputs_dir=None, save_outputs=False, *,
               amp=True, lpips_backbone="alex", is_splits=10,
               return_per_image=False):

    Enc1.eval()
    Dec2.eval()

    metric_sums = {
        "mse": 0.0,
        "psnr": 0.0,
        "vif": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }
    per_image = {k: [] for k in ["mse","psnr","vif","ssim","lpips"]}

    n_images = 0

    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    inc_cls = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat.fc = nn.Identity()

    logits_all, feats_real_all, feats_fake_all = [], [], []

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for batch_idx, batch in enumerate(pbar):
        x1, x2, file_names = batch
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)

        bsz = x1.size(0)
        n_images += bsz

        # forward
        with scaler_ctx():
            content, style = Enc1(x1)
            fake_x2 = Dec2(content, style)

        # ---- Tanh [-1,1] -> [0,1] ----
        fake_x2_01 = (fake_x2 + 1.0) / 2.0
        fake_x2_01 = fake_x2_01.clamp(0, 1)

        # salva output
        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_x2_01, file_names):
                save_image(out_img, os.path.join(outputs_dir, f"{name}_output.png"))


        mse_val  = _to_scalar(mse_metric(fake_x2_01, x2))
        psnr_val = _to_scalar(psnr_metric(fake_x2_01, x2))

        with torch.cuda.amp.autocast(enabled=False):
            vif_val  = _to_scalar(piq_vif(fake_x2_01.float(), x2.float()))
            ssim_val = _to_scalar(tm_ssim(fake_x2_01, x2, data_range=1.0))

        a_lp = _prep_for_lpips(fake_x2_01)
        b_lp = _prep_for_lpips(x2)
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

        # Inception
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fake_inc = _prep_for_inception(fake_x2_01).float().to(device)
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

    # FID
    if feats_fake_all and feats_real_all:
        Ffake = np.concatenate(feats_fake_all, axis=0)
        Freal = np.concatenate(feats_real_all, axis=0)
        mu_f, sig_f = Ffake.mean(axis=0), np.cov(Ffake, rowvar=False)
        mu_r, sig_r = Freal.mean(axis=0), np.cov(Freal, rowvar=False)
        results["fid"] = _frechet_distance(mu_r, sig_r, mu_f, sig_f)
    else:
        results["fid"] = float("nan")

    # IS
    if logits_all:
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



# ---------------- helper ----------------

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
    return (x - mean) / std

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
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
    probs = torch.softmax(logits, dim=1)
    split_scores = []
    for k in range(splits):
        part = probs[k*(N//splits):(k+1)*(N//splits)]
        py = part.mean(dim=0, keepdim=True)
        kl = (part * (part.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(dim=1)
        split_scores.append(torch.exp(kl.mean()).item())
    return float(np.mean(split_scores)), float(np.std(split_scores))

