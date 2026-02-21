import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchmetrics import PeakSignalNoiseRatio
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from src.utils.util_general import save_checkpoint
import src.utils.util_general as util_general
import matplotlib.pyplot as plt
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

# ---------------------------
# TRAIN ONE EPOCH
# ---------------------------
def train_fn(
    G, D,
    loader,
    opt_G, opt_D,
    mse_gan, l1,
    lambdas, device,
    g_scaler, d_scaler
):
    G.train(); D.train()
    loop = tqdm(loader, leave=True)

    for xA, xB, _ in loop:
        xA = xA.to(device, non_blocking=True).float()  # MRI
        xB = xB.to(device, non_blocking=True).float()  # CT

        # ------------------
        # Train Generator
        # ------------------
        opt_G.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            # forward
            fake_B = G(xA, to="B")
            fake_A = G(xB, to="A")

            # cycle
            cyc_A = G(fake_B, to="A")
            cyc_B = G(fake_A, to="B")

            # identity / recon
            id_A = G(xA, to="A")
            id_B = G(xB, to="B")

            # adv losses (LSGAN)
            pred_fake_A = D(fake_A, domain="A")
            pred_fake_B = D(fake_B, domain="B")
            loss_gan_A = mse_gan(pred_fake_A, torch.ones_like(pred_fake_A))
            loss_gan_B = mse_gan(pred_fake_B, torch.ones_like(pred_fake_B))

            # cycle & identity
            loss_cyc = l1(cyc_A, xA) + l1(cyc_B, xB)
            loss_id  = l1(id_A, xA) + l1(id_B, xB)

            LG = (
                lambdas["lambda_gan"] * (loss_gan_A + loss_gan_B)
                + lambdas["lambda_cyc"] * loss_cyc
                + lambdas["lambda_id"]  * loss_id
            )

        g_scaler.scale(LG).backward()
        g_scaler.step(opt_G)
        g_scaler.update()

        # ------------------
        # Train Discriminator
        # ------------------
        opt_D.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            # real
            pred_real_A = D(xA, domain="A")
            pred_real_B = D(xB, domain="B")
            loss_D_real_A = mse_gan(pred_real_A, torch.ones_like(pred_real_A))
            loss_D_real_B = mse_gan(pred_real_B, torch.ones_like(pred_real_B))

            # fake
            pred_fake_A = D(fake_A.detach(), domain="A")
            pred_fake_B = D(fake_B.detach(), domain="B")
            loss_D_fake_A = mse_gan(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_D_fake_B = mse_gan(pred_fake_B, torch.zeros_like(pred_fake_B))

            LD = 0.5 * (loss_D_real_A + loss_D_fake_A + loss_D_real_B + loss_D_fake_B)

        d_scaler.scale(LD).backward()
        d_scaler.step(opt_D)
        d_scaler.update()

    return G, D


# ---------------------------
# EVAL (L1 solo MRI->CT)
# ---------------------------
@torch.no_grad()
def eval_fn(G, loader, criterion, device):
    G.eval()
    tot = 0.0
    n = 0
    loop = tqdm(loader, leave=False)
    for xA, xB, _ in loop:
        xA = xA.to(device).float()
        xB = xB.to(device).float()
        with torch.cuda.amp.autocast():
            fake_B = G(xA, to="B")
        loss = criterion(fake_B, xB)
        tot += loss.item() * xA.size(0)
        n += xA.size(0)
    return tot / max(1, n)


def train_cogan(
    G, D,
    data_loaders, early_stopping_criterion,
    opt_G, opt_D,
    l1, mse_gan,
    model_fold_dir, cfg_trainer, device, resume_state=None
):
    lambdas = dict(
        lambda_gan = cfg_trainer.get("lambda_gan", 1.0),
        lambda_cyc = cfg_trainer.get("lambda_cyc", 10.0),
        lambda_id  = cfg_trainer.get("lambda_id", 5.0),
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best_wts = copy.deepcopy(G.state_dict())
    best_loss = np.Inf
    best_epoch = -1
    hist = {"train_loss": [], "val_loss": []}
    epochs_no_improve = 0
    early_stop = False

    max_epochs = cfg_trainer["max_epochs"]
    warm_up    = cfg_trainer.get("warm_up", 50)
    patience   = cfg_trainer.get("early_stopping", 50)

    ckpt_every = cfg_trainer.get("ckpt_every", 50)

    for epoch in range(max_epochs):
        # TRAIN
        G, D = train_fn(
            G, D,
            data_loaders["train"],
            opt_G, opt_D,
            mse_gan, l1,
            lambdas, device,
            g_scaler, d_scaler
        )

        # EVAL
        val_loss = eval_fn(G, data_loaders["val"], early_stopping_criterion, device)
        train_loss = eval_fn(G, data_loaders["train"], early_stopping_criterion, device)

        hist["val_loss"].append(val_loss)
        hist["train_loss"].append(train_loss)

        # Checkpoint overwrite ogni ckpt_every epoche nella stessa cartella
        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(G, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G"]))
            save_checkpoint(D, opt_D, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D"]))
            print(f"[Checkpoint] overwrite a epoca {epoch+1} in {model_fold_dir}")

        # Early stopping dopo warmup
        if epoch > warm_up:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_wts = copy.deepcopy(G.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly Stopping @ epoch {epoch} (best={best_epoch}, val={best_loss:.6f})")
                    early_stop = True

        print(f"[Epoch {epoch+1}/{max_epochs}] train_L1(MRI->CT)={train_loss:.6f} | val_L1(MRI->CT)={val_loss:.6f}")

        if early_stop:
            break

    # carico best weights
    G.load_state_dict(best_wts)

    # salvataggio finale (stessi nomi, overwrite definitivo)
    save_checkpoint(G, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G"]))
    save_checkpoint(D, opt_D, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D"]))

    history = pd.DataFrame.from_dict(hist, orient="index").transpose()
    return G, D, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 MRI->CT')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


# ---------------------------
# TEST metrics MRI->CT
# ---------------------------
@torch.no_grad()
def test_4metrics_cogan(G, loader_test, device, outputs_dir=None, save_output=True):
    G.eval()
    mse_metric = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    run_mse = run_psnr = run_vif = run_ssim = 0.0
    n = 0

    loop = tqdm(loader_test, leave=True)
    for xA, xB, file_names in loop:
        xA = xA.to(device).float()
        xB = xB.to(device).float()

        with torch.cuda.amp.autocast():
            fake_B = G(xA, to="B")

        if save_output and outputs_dir:
            for out_img, name in zip(fake_B, file_names):
                save_image(out_img, os.path.join(outputs_dir, f"{name}_cogan_output.png"))

        v_mse  = mse_metric(fake_B, xB).item()
        v_psnr = psnr_metric(fake_B, xB).item()
        v_vif  = vif_p(fake_B.float(), xB.float()).item()
        v_ssim = ssim_fn(fake_B.float(), xB.float(), data_range=1.0).item()

        bsz = xA.size(0)
        run_mse  += v_mse * bsz
        run_psnr += v_psnr * bsz
        run_vif  += v_vif * bsz
        run_ssim += v_ssim * bsz
        n += bsz

    return (run_mse / n, run_psnr / n, run_vif / n, run_ssim / n)
@torch.no_grad()
def test_cogan(G, loader_test, mse_metric, psnr_metric, device,
               outputs_dir=None, save_outputs=False, *,
               amp=True, lpips_backbone="alex", is_splits=10,
               return_per_image=False):

    G.eval()

    metric_sums = {"mse":0.0,"psnr":0.0,"vif":0.0,"ssim":0.0,"lpips":0.0}
    per_image = {"mse":[], "psnr":[], "vif":[], "ssim":[], "lpips":[]}
    n_images = 0

    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

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
        bsz = x1.size(0); n_images += bsz

        with scaler_ctx():
            out = G(x1)
            # CoGAN spesso ritorna (fake_A, fake_B)
            fake_x2 = out[1] if isinstance(out, (tuple, list)) else out

        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img.clamp(0,1), os.path.join(outputs_dir, f"{name}_output.png"))

        mse_val  = _to_scalar(mse_metric(fake_x2, x2))
        psnr_val = _to_scalar(psnr_metric(fake_x2, x2))

        with torch.cuda.amp.autocast(enabled=False):
            vif_val  = _to_scalar(piq_vif(fake_x2.float(), x2.float()))
            ssim_val = _to_scalar(tm_ssim(fake_x2, x2, data_range=1.0))

        a_lp = _prep_for_lpips(fake_x2)
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
    for k,v in metric_sums.items():
        results[k] = v/denom

    # FID
    if len(feats_fake_all)>0 and len(feats_real_all)>0:
        Ffake = np.concatenate(feats_fake_all, axis=0)
        Freal = np.concatenate(feats_real_all, axis=0)
        mu_f, sig_f = Ffake.mean(axis=0), np.cov(Ffake, rowvar=False)
        mu_r, sig_r = Freal.mean(axis=0), np.cov(Freal, rowvar=False)
        results["fid"] = _frechet_distance(mu_r, sig_r, mu_f, sig_f)
    else:
        results["fid"] = float("nan")

    # IS
    if len(logits_all)>0:
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