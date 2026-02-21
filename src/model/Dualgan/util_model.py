import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.autograd as autograd
from torchvision.utils import save_image
from torchmetrics import PeakSignalNoiseRatio
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from src.utils.util_general import save_checkpoint
import src.utils.util_general as util_general
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
# Gradient Penalty WGAN-GP
# ---------------------------
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones_like(d_interpolates, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ---------------------------
# Train 1 epoch
# ---------------------------
def train_fn(
    G_AB, G_BA, D_A, D_B,
    loader,
    opt_G, opt_D_A, opt_D_B,
    cycle_loss,
    cfg_trainer, device,
    g_scaler, dA_scaler, dB_scaler
):
    G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

    lambda_adv   = cfg_trainer.get("lambda_adv", 1.0)
    lambda_cycle = cfg_trainer.get("lambda_cycle", 10.0)
    lambda_gp    = cfg_trainer.get("lambda_gp", 10.0)

    loop = tqdm(loader, leave=True)
    for i, (img_A, img_B, _) in enumerate(loop):
        img_A = img_A.to(device, non_blocking=True).float()
        img_B = img_B.to(device, non_blocking=True).float()

        # ----------------------
        # Train Discriminators
        # ----------------------
        # Fake
        with torch.cuda.amp.autocast():
            fake_A = G_BA(img_B).detach()
            fake_B = G_AB(img_A).detach()

        # D_A
        opt_D_A.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            gp_A = compute_gradient_penalty(D_A, img_A, fake_A, device)
            D_A_loss = -torch.mean(D_A(img_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A
        dA_scaler.scale(D_A_loss).backward()
        dA_scaler.step(opt_D_A)
        dA_scaler.update()

        # D_B
        opt_D_B.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            gp_B = compute_gradient_penalty(D_B, img_B, fake_B, device)
            D_B_loss = -torch.mean(D_B(img_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B
        dB_scaler.scale(D_B_loss).backward()
        dB_scaler.step(opt_D_B)
        dB_scaler.update()

        # ----------------------
        # Train Generators (n_critic)
        # ----------------------
        n_critic = cfg_trainer.get("n_critic", 5)
        if i % n_critic == 0:
            opt_G.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                fake_A = G_BA(img_B)
                fake_B = G_AB(img_A)

                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)

                G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
                G_cyc = cycle_loss(recov_A, img_A) + cycle_loss(recov_B, img_B)

                G_loss = lambda_adv * G_adv + lambda_cycle * G_cyc

            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_G)
            g_scaler.update()

        loop.set_postfix(
            D_A=D_A_loss.item(),
            D_B=D_B_loss.item(),
            G=G_loss.item() if i % n_critic == 0 else None
        )

    return G_AB, G_BA, D_A, D_B


# ---------------------------
# Eval L1 MRI->CT (A->B)
# ---------------------------
@torch.no_grad()
def eval_fn(G_AB, loader, criterion, device):
    G_AB.eval()
    tot = 0.0
    n = 0
    loop = tqdm(loader, leave=False)
    for img_A, img_B, _ in loop:
        img_A = img_A.to(device).float()
        img_B = img_B.to(device).float()
        with torch.cuda.amp.autocast():
            fake_B = G_AB(img_A)
        loss = criterion(fake_B, img_B)
        bs = img_A.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / max(1, n)


# ---------------------------
# Main train with early stopping + ckpt overwrite
# ---------------------------
def train_dualgan(
    G_AB, G_BA, D_A, D_B,
    data_loaders,
    opt_G, opt_D_A, opt_D_B,
    cycle_loss,
    cfg_trainer,
    model_fold_dir,
    device
):
    early_stopping_criterion = nn.L1Loss().to(device)

    g_scaler  = torch.cuda.amp.GradScaler()
    dA_scaler = torch.cuda.amp.GradScaler()
    dB_scaler = torch.cuda.amp.GradScaler()

    best_wts = {
        "G_AB": copy.deepcopy(G_AB.state_dict()),
        "G_BA": copy.deepcopy(G_BA.state_dict()),
        "D_A":  copy.deepcopy(D_A.state_dict()),
        "D_B":  copy.deepcopy(D_B.state_dict()),
    }

    best_loss = np.inf
    best_epoch = -1
    hist = {"train_loss": [], "val_loss": []}
    epochs_no_improve = 0
    early_stop = False

    max_epochs = cfg_trainer.get("max_epochs", 300)
    warm_up    = cfg_trainer.get("warm_up", 0)
    patience   = cfg_trainer.get("early_stopping", 50)
    ckpt_every = cfg_trainer.get("ckpt_every", 50)

    for epoch in range(max_epochs):
        # TRAIN
        G_AB, G_BA, D_A, D_B = train_fn(
            G_AB, G_BA, D_A, D_B,
            data_loaders["train"],
            opt_G, opt_D_A, opt_D_B,
            cycle_loss,
            cfg_trainer, device,
            g_scaler, dA_scaler, dB_scaler
        )

        # EVAL
        train_loss = eval_fn(G_AB, data_loaders["train"], early_stopping_criterion, device)
        val_loss   = eval_fn(G_AB, data_loaders["val"], early_stopping_criterion, device)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)


        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(G_AB, opt_G,  filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G_AB"]))
            save_checkpoint(G_BA, opt_G,  filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G_BA"]))
            save_checkpoint(D_A,  opt_D_A, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D_A"]))
            save_checkpoint(D_B,  opt_D_B, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D_B"]))
            print(f"[Checkpoint] overwrite a epoca {epoch+1} in {model_fold_dir}")

        # Early stopping after warmup
        if epoch > warm_up:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_wts = {
                    "G_AB": copy.deepcopy(G_AB.state_dict()),
                    "G_BA": copy.deepcopy(G_BA.state_dict()),
                    "D_A":  copy.deepcopy(D_A.state_dict()),
                    "D_B":  copy.deepcopy(D_B.state_dict()),
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly Stopping @ epoch {epoch} (best={best_epoch}, val={best_loss:.6f})")
                    early_stop = True

        print(f"[Epoch {epoch+1}/{max_epochs}] train_L1(A->B)={train_loss:.6f} | val_L1(A->B)={val_loss:.6f}")

        if early_stop:
            break

    # load best weights
    G_AB.load_state_dict(best_wts["G_AB"])
    G_BA.load_state_dict(best_wts["G_BA"])
    D_A.load_state_dict(best_wts["D_A"])
    D_B.load_state_dict(best_wts["D_B"])

    # save final best
    save_checkpoint(G_AB, opt_G,  filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G_AB"]))
    save_checkpoint(G_BA, opt_G,  filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G_BA"]))
    save_checkpoint(D_A,  opt_D_A, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D_A"]))
    save_checkpoint(D_B,  opt_D_B, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D_B"]))

    history = pd.DataFrame.from_dict(hist, orient="index").transpose()
    return G_AB, G_BA, D_A, D_B, history


# ---------------------------
# Plot training
# ---------------------------
def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Training L1 (A->B)')
    plt.plot(history['val_loss'], label='Validation L1 (A->B)')
    plt.xlabel('Epoch')
    plt.ylabel('L1')
    plt.title('DualGAN Training / Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


# ---------------------------
# Test 4 metrics (A->B)
# ---------------------------
@torch.no_grad()
def test_4metrics(
    G_AB, loader_test, device,
    outputs_dir=None, save_output=True
):
    G_AB.eval()
    mse_metric  = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    run_mse = run_psnr = run_vif = run_ssim = 0.0
    n = 0

    loop = tqdm(loader_test, leave=True)
    for img_A, img_B, file_names in loop:
        img_A = img_A.to(device).float()
        img_B = img_B.to(device).float()

        with torch.cuda.amp.autocast():
            fake_B = G_AB(img_A)

        # save png
        if save_output and outputs_dir:
            for out_img, name in zip(fake_B, file_names):
                save_image(out_img, os.path.join(outputs_dir, f"{name}_dualgan_output.png"))

        v_mse  = mse_metric(fake_B, img_B).item()
        v_psnr = psnr_metric(fake_B, img_B).item()
        v_vif  = vif_p(fake_B.float(), img_B.float()).item()
        v_ssim = ssim_fn(fake_B.float(), img_B.float(), data_range=1.0).item()

        bs = img_A.size(0)
        run_mse  += v_mse  * bs
        run_psnr += v_psnr * bs
        run_vif  += v_vif  * bs
        run_ssim += v_ssim * bs
        n += bs

    return (run_mse / n, run_psnr / n, run_vif / n, run_ssim / n)


def test_dualgan(G_AB, loader_test, mse_metric, psnr_metric, device,
                 outputs_dir=None, save_outputs=False, *,
                 amp=True, lpips_backbone="alex", is_splits=10,
                 return_per_image=False):
    G_AB.eval()

    metric_sums = {
        "mse": 0.0,
        "psnr": 0.0,
        "vif": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }
    per_image = {k: [] for k in metric_sums.keys()}
    n_images = 0

    # LPIPS network
    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    # Inception for IS/FID
    inc_cls = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat.fc = nn.Identity()  # features 2048

    logits_all = []
    feats_real_all = []
    feats_fake_all = []

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for batch_idx, batch in enumerate(pbar):
        img_A, img_B, file_names = batch
        img_A = img_A.to(device, dtype=torch.float32)
        img_B = img_B.to(device, dtype=torch.float32)

        bsz = img_A.size(0)
        n_images += bsz

        # forward
        with scaler_ctx():
            fake_B = G_AB(img_A)

        # save outputs (assume [0,1])
        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_B, file_names):
                save_image(out_img.clamp(0, 1),
                           os.path.join(outputs_dir, f"{name}_dualgan_output.png"))

        # base metrics
        mse_val = _to_scalar(mse_metric(fake_B, img_B))
        psnr_val = _to_scalar(psnr_metric(fake_B, img_B))

        with torch.cuda.amp.autocast(enabled=False):
            vif_val = _to_scalar(piq_vif(fake_B.float(), img_B.float()))
            ssim_val = _to_scalar(tm_ssim(fake_B, img_B, data_range=1.0))

        # LPIPS wants [-1,1] and 3ch
        a_lp = _prep_for_lpips(fake_B)
        b_lp = _prep_for_lpips(img_B)
        lpips_val = _to_scalar(lpips_fn(a_lp, b_lp).mean())

        # weighted sums
        metric_sums["mse"] += mse_val * bsz
        metric_sums["psnr"] += psnr_val * bsz
        metric_sums["vif"] += vif_val * bsz
        metric_sums["ssim"] += ssim_val * bsz
        metric_sums["lpips"] += lpips_val * bsz

        if return_per_image:
            per_image["mse"].append(mse_val)
            per_image["psnr"].append(psnr_val)
            per_image["vif"].append(vif_val)
            per_image["ssim"].append(ssim_val)
            per_image["lpips"].append(lpips_val)

        # Inception stuff
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fake_inc = _prep_for_inception(fake_B).float().to(device)
                real_inc = _prep_for_inception(img_B).float().to(device)

                logits = inc_cls(fake_inc)
                logits_all.append(logits.detach().cpu())

                f_fake = inc_feat(fake_inc)
                f_real = inc_feat(real_inc)
                feats_fake_all.append(f_fake.detach().cpu().numpy())
                feats_real_all.append(f_real.detach().cpu().numpy())

    # averages
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

    # IS
    if len(logits_all) > 0:
        logits_cat = torch.cat(logits_all, dim=0)
        is_mean, is_std = _inception_score_from_logits(logits_cat, splits=is_splits)
        results["is_mean"] = float(is_mean)
        results["is_std"] = float(is_std)
    else:
        results["is_mean"] = float("nan")
        results["is_std"] = float("nan")

    if return_per_image:
        results["per_image"] = per_image

    return results

    # ----------------- helpers -----------------


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
    return x * 2.0 - 1.0


def _prep_for_inception(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_3ch(x.clamp(0, 1))
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1 = np.atleast_1d(mu1);
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1);
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
    probs = torch.softmax(logits, dim=1)
    split_scores = []
    for k in range(splits):
        part = probs[k * (N // splits):(k + 1) * (N // splits)]
        py = part.mean(dim=0, keepdim=True)
        kl = (part * (part.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(dim=1)
        split_scores.append(torch.exp(kl.mean()).item())
    return float(np.mean(split_scores)), float(np.std(split_scores))