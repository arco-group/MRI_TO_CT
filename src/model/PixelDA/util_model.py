import os, copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchmetrics import PeakSignalNoiseRatio
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

import src.utils.util_general as util_general
from src.utils.util_general import save_checkpoint
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

# -------------------------
# Train one epoch
# -------------------------
def train_fn(
    G, D, C,
    loader,
    opt_GC, opt_D,
    adv_loss, task_loss,
    lambda_adv, lambda_task,
    latent_dim,
    device,
    scaler_GC, scaler_D
):
    G.train(); D.train(); C.train()
    loop = tqdm(loader, leave=True)

    run_g = run_d = run_task = run_adv = 0.0
    n = 0

    for mri, ct, y, _ in loop:
        mri = mri.to(device, non_blocking=True).float()
        ct  = ct.to(device, non_blocking=True).float()
        y   = y.to(device, non_blocking=True).long()

        bs = mri.size(0)
        n += bs

        valid = torch.ones((bs, 1, ct.size(2)//16, ct.size(3)//16), device=device)
        fake  = torch.zeros_like(valid)

        # -----------------
        #  Train G + C
        # -----------------
        opt_GC.zero_grad(set_to_none=True)
        z = torch.randn(bs, latent_dim, device=device)

        with torch.cuda.amp.autocast():
            fake_ct = G(mri, z)

            pred_fake = D(fake_ct)
            loss_adv = adv_loss(pred_fake, valid)


            logits_fake = C(fake_ct)
            logits_mri  = C(mri)
            loss_task = (task_loss(logits_fake, y) + task_loss(logits_mri, y)) / 2.0

            loss_GC = lambda_adv * loss_adv + lambda_task * loss_task

        scaler_GC.scale(loss_GC).backward()
        scaler_GC.step(opt_GC)
        scaler_GC.update()

        # -----------------
        #  Train D
        # -----------------
        opt_D.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred_real = D(ct)
            pred_fake = D(fake_ct.detach())
            loss_D = (adv_loss(pred_real, valid) + adv_loss(pred_fake, fake)) / 2.0

        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        run_g += loss_GC.item() * bs
        run_d += loss_D.item()  * bs
        run_task += loss_task.item() * bs
        run_adv  += loss_adv.item()  * bs

        loop.set_postfix(G=loss_GC.item(), D=loss_D.item(), task=loss_task.item())

    return (run_g/n, run_d/n, run_adv/n, run_task/n)


@torch.no_grad()
def eval_fn(G, D, C, loader, adv_loss, task_loss, lambda_adv, lambda_task, latent_dim, device):
    G.eval(); D.eval(); C.eval()
    loop = tqdm(loader, leave=False)

    run_g = run_adv = run_task = 0.0
    correct = total = 0
    n = 0

    for mri, ct, y, _ in loop:
        mri = mri.to(device).float()
        ct  = ct.to(device).float()
        y   = y.to(device).long()
        bs = mri.size(0)
        n += bs

        valid = torch.ones((bs, 1, ct.size(2)//16, ct.size(3)//16), device=device)

        z = torch.randn(bs, latent_dim, device=device)
        fake_ct = G(mri, z)

        pred_fake = D(fake_ct)
        loss_adv = adv_loss(pred_fake, valid)

        logits_fake = C(fake_ct)
        loss_task = task_loss(logits_fake, y)

        loss_G = lambda_adv * loss_adv + lambda_task * loss_task

        run_g += loss_G.item() * bs
        run_adv += loss_adv.item() * bs
        run_task += loss_task.item() * bs

        pred = logits_fake.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += bs

    acc = correct / max(1, total)
    return (run_g/n, run_adv/n, run_task/n, acc)


def train_pixelda(
    G, D, C,
    data_loaders,
    opt_GC, opt_D,
    adv_loss, task_loss,
    cfg_trainer,
    model_fold_dir,
    device
):
    lambda_adv  = cfg_trainer.get("lambda_adv", 1.0)
    lambda_task = cfg_trainer.get("lambda_task", 0.1)
    latent_dim  = cfg_trainer.get("latent_dim", 8)

    max_epochs = cfg_trainer["max_epochs"]
    warm_up    = cfg_trainer.get("warm_up", 0)
    patience   = cfg_trainer.get("early_stopping", 50)
    ckpt_every = cfg_trainer.get("ckpt_every", 50)

    scaler_GC = torch.cuda.amp.GradScaler()
    scaler_D  = torch.cuda.amp.GradScaler()

    best_wts = {
        "G": copy.deepcopy(G.state_dict()),
        "D": copy.deepcopy(D.state_dict()),
        "C": copy.deepcopy(C.state_dict())
    }
    best_loss = np.Inf
    best_epoch = -1
    no_improve = 0

    hist = {
        "train_G": [], "train_D": [], "train_adv": [], "train_task": [],
        "val_G": [], "val_adv": [], "val_task": [], "val_acc": []
    }

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        tr_G, tr_D, tr_adv, tr_task = train_fn(
            G, D, C,
            data_loaders["train"],
            opt_GC, opt_D,
            adv_loss, task_loss,
            lambda_adv, lambda_task,
            latent_dim,
            device,
            scaler_GC, scaler_D
        )

        val_G, val_adv, val_task, val_acc = eval_fn(
            G, D, C,
            data_loaders["val"],
            adv_loss, task_loss,
            lambda_adv, lambda_task,
            latent_dim,
            device
        )

        hist["train_G"].append(tr_G)
        hist["train_D"].append(tr_D)
        hist["train_adv"].append(tr_adv)
        hist["train_task"].append(tr_task)

        hist["val_G"].append(val_G)
        hist["val_adv"].append(val_adv)
        hist["val_task"].append(val_task)
        hist["val_acc"].append(val_acc)

        print(f"Train: G={tr_G:.4f} D={tr_D:.4f} adv={tr_adv:.4f} task={tr_task:.4f}")
        print(f"Val:   G={val_G:.4f} adv={val_adv:.4f} task={val_task:.4f} acc={val_acc*100:.1f}%")


        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(G, opt_GC, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G"]))
            save_checkpoint(D, opt_D,  filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D"]))
            save_checkpoint(C, opt_GC, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_C"]))
            print(f"[Checkpoint overwrite] epoca {epoch+1} salvata in {model_fold_dir}")


        if epoch > warm_up:
            if val_G < best_loss:
                best_loss = val_G
                best_epoch = epoch
                no_improve = 0
                best_wts = {
                    "G": copy.deepcopy(G.state_dict()),
                    "D": copy.deepcopy(D.state_dict()),
                    "C": copy.deepcopy(C.state_dict())
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping @ epoch {epoch+1}")
                    break

    print(f"Best epoch: {best_epoch+1}  best val G loss: {best_loss:.6f}")


    G.load_state_dict(best_wts["G"])
    D.load_state_dict(best_wts["D"])
    C.load_state_dict(best_wts["C"])


    save_checkpoint(G, opt_GC, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G"]))
    save_checkpoint(D, opt_D,  filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D"]))
    save_checkpoint(C, opt_GC, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_C"]))

    history = pd.DataFrame(hist)
    return G, D, C, history

import os
import matplotlib.pyplot as plt
import pandas as pd
import src.utils.util_general as util_general


def plot_training(history, model_name, plot_training_dir):

    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)


    if isinstance(history, dict):
        history = pd.DataFrame(history)
    elif not isinstance(history, pd.DataFrame):

        history = pd.DataFrame(history)

    if history.empty:
        print("[plot_training] history vuota, nessun plot salvato.")
        return


    def _plot_cols(train_col, val_col, title, ylab, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(history[train_col], label=train_col)
        plt.plot(history[val_col], label=val_col)
        plt.xlabel("Epoch")
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_plot_dir, filename))
        plt.close()


    cols = set(history.columns)


    candidate_pairs = []
    for c in cols:
        if c.startswith("train_"):
            v = "val_" + c[len("train_"):]
            if v in cols:
                candidate_pairs.append((c, v))


    if not candidate_pairs:
        if "train_loss" in cols and "val_loss" in cols:
            candidate_pairs.append(("train_loss", "val_loss"))
        if "train_acc" in cols and "val_acc" in cols:
            candidate_pairs.append(("train_acc", "val_acc"))

    if not candidate_pairs:
        print(f"[plot_training] No copy train/val in history: {list(cols)}")
        return


    for tr_col, va_col in candidate_pairs:
        metric_name = tr_col.replace("train_", "")
        title = f"{model_name} - {metric_name}"
        ylab = metric_name
        filename = f"{metric_name}.png"

        _plot_cols(tr_col, va_col, title, ylab, filename)

    print(f"[plot_training] Salvati {len(candidate_pairs)} plot in {model_plot_dir}")


@torch.no_grad()
def test_4metrics_pixelda(G, C, loader_test, device, outputs_dir=None, save_output=True):
    from torch.nn import MSELoss

    G.eval(); C.eval()

    mse_metric = MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    run_mse = run_psnr = run_vif = run_ssim = 0.0
    correct = total = 0
    n = 0

    loop = tqdm(loader_test, leave=True)
    for mri, ct, y, file_names in loop:
        mri = mri.to(device).float()
        ct  = ct.to(device).float()
        y   = y.to(device).long()

        bs = mri.size(0)
        z = torch.randn(bs, G.latent_dim, device=device)
        fake_ct = G(mri, z)

        if save_output and outputs_dir:
            for out_img, name in zip(fake_ct, file_names):
                save_image(out_img, os.path.join(outputs_dir, f"{name}_pixelda_ct.png"))

        v_mse  = mse_metric(fake_ct, ct).item()
        v_psnr = psnr_metric(fake_ct, ct).item()
        v_vif  = vif_p(fake_ct, ct, data_range=1.0).item()
        v_ssim = ssim_fn(fake_ct, ct, data_range=1.0).item()

        logits = C(fake_ct)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += bs

        run_mse  += v_mse  * bs
        run_psnr += v_psnr * bs
        run_vif  += v_vif  * bs
        run_ssim += v_ssim * bs
        n += bs

    acc = correct / max(1, total)
    return (run_mse/n, run_psnr/n, run_vif/n, run_ssim/n, acc)
@torch.no_grad()
def test_pixelda(G, loader_test, mse_metric, psnr_metric,
                 latent_dim, device,
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
            fake_x2 = _forward_pixelda(G, x1, latent_dim, device)

        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img.clamp(0,1),
                           os.path.join(outputs_dir, f"{name}_output.png"))

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

        # ---- Inception (IS + FID) ----
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


def _forward_pixelda(G, x, latent_dim, device):

    try:
        return G(x)
    except TypeError:
        z = torch.randn(x.size(0), latent_dim, device=device)
        return G(x, z)


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