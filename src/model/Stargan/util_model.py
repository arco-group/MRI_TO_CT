import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import save_image

from piq import vif_p
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure

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

def criterion_cls(logit, target):

    return F.binary_cross_entropy_with_logits(logit, target, reduction="sum") / logit.size(0)


def compute_gradient_penalty(D, real_samples, fake_samples, device):

    Tensor = torch.cuda.FloatTensor if real_samples.is_cuda else torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_fn(
    generator,
    discriminator,
    loader,
    opt_G,
    opt_D,
    l1_cycle,
    lambdas,
    n_critic,
    device,
    g_scaler: torch.cuda.amp.GradScaler,
    d_scaler: torch.cuda.amp.GradScaler,
):

    generator.train()
    discriminator.train()

    lambda_cls = lambdas["lambda_cls"]
    lambda_rec = lambdas["lambda_rec"]
    lambda_gp = lambdas["lambda_gp"]

    loop = tqdm(loader, leave=True)
    running_loss_G = 0.0
    running_loss_D = 0.0
    steps_G = 0

    for i, (mri, ct, _) in enumerate(loop):
        mri = mri.to(device)  # [B,1,H,W]
        ct = ct.to(device)    # [B,1,H,W]

        # batch
        x = torch.cat([mri, ct], dim=0)  # [2B,1,H,W]
        b = mri.size(0)


        c_org = torch.zeros(x.size(0), 2, device=device)
        c_org[:b, 0] = 1.0  # MRI
        c_org[b:, 1] = 1.0  # CT


        c_trg = torch.zeros_like(c_org)
        c_trg[:b, 1] = 1.0  # MRI -> CT
        c_trg[b:, 0] = 1.0  # CT -> MRI

        # ------------------------------------------------
        # Train Discriminator
        # ------------------------------------------------
        opt_D.zero_grad()
        with torch.cuda.amp.autocast():
            # real
            out_src_real, out_cls_real = discriminator(x)
            # fake
            x_fake = generator(x, c_trg)
            out_src_fake, _ = discriminator(x_fake.detach())

            # GP
            gp = compute_gradient_penalty(discriminator, x.data, x_fake.data, device)

            # adversarial
            loss_D_adv = -torch.mean(out_src_real) + torch.mean(out_src_fake) + lambda_gp * gp
            # classification on real
            loss_D_cls = criterion_cls(out_cls_real, c_org)

            loss_D = loss_D_adv + lambda_cls * loss_D_cls

        d_scaler.scale(loss_D).backward()
        d_scaler.step(opt_D)
        d_scaler.update()

        running_loss_D += loss_D.item() * x.size(0)

        # ------------------------------------------------
        # Train Generator
        # ------------------------------------------------
        if i % n_critic == 0:
            opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                x_fake = generator(x, c_trg)
                out_src_fake, out_cls_fake = discriminator(x_fake)


                x_rec = generator(x_fake, c_org)

                loss_G_adv = -torch.mean(out_src_fake)
                loss_G_cls = criterion_cls(out_cls_fake, c_trg)
                loss_G_rec = l1_cycle(x_rec, x)

                loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec

            g_scaler.scale(loss_G).backward()
            g_scaler.step(opt_G)
            g_scaler.update()

            running_loss_G += loss_G.item() * x.size(0)
            steps_G += 1

            loop.set_postfix(D=loss_D.item(), G=loss_G.item(), rec=loss_G_rec.item())

    n_samples = len(loader.dataset) * 2
    epoch_loss_D = running_loss_D / n_samples
    epoch_loss_G = running_loss_G / max(steps_G, 1)
    return generator, discriminator, epoch_loss_G, epoch_loss_D


def eval_fn(generator, loader_val, l1_cycle, device):

    generator.eval()
    running_loss = 0.0
    loop = tqdm(loader_val, leave=False)

    with torch.no_grad():
        for mri, ct, _ in loop:
            mri = mri.to(device)
            ct = ct.to(device)

            c_mri = torch.tensor([[1.0, 0.0]], device=device).repeat(mri.size(0), 1)
            c_ct  = torch.tensor([[0.0, 1.0]], device=device).repeat(mri.size(0), 1)

            with torch.cuda.amp.autocast():
                fake_ct = generator(mri, c_ct)
                loss = l1_cycle(fake_ct, ct)

            running_loss += loss.item() * mri.size(0)

    return running_loss / len(loader_val.dataset)


def train_stargan(
    generator,
    discriminator,
    data_loaders,
    opt_G,
    opt_D,
    l1_cycle,
    lambdas,
    n_critic,
    model_fold_dir,
    cfg_trainer,
    device,
):

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best_G_w = copy.deepcopy(generator.state_dict())
    best_D_w = copy.deepcopy(discriminator.state_dict())

    best_loss = np.Inf
    best_epoch = 0
    history = {"train_loss_G": [], "train_loss_D": [], "val_loss": []}
    epochs_no_improve = 0
    early_stop = False
    warm_up = cfg_trainer.get("warm_up", 0)
    early_stopping = cfg_trainer.get("early_stopping", 0)

    max_epochs = cfg_trainer["max_epochs"]

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        generator, discriminator, train_G, train_D = train_fn(
            generator=generator,
            discriminator=discriminator,
            loader=data_loaders["train"],
            opt_G=opt_G,
            opt_D=opt_D,
            l1_cycle=l1_cycle,
            lambdas=lambdas,
            n_critic=n_critic,
            device=device,
            g_scaler=g_scaler,
            d_scaler=d_scaler,
        )

        val_loss = eval_fn(
            generator=generator,
            loader_val=data_loaders["val"],
            l1_cycle=l1_cycle,
            device=device,
        )

        history["train_loss_G"].append(train_G)
        history["train_loss_D"].append(train_D)
        history["val_loss"].append(val_loss)

        print(f"Train G: {train_G:.4f}  Train D: {train_D:.4f}  Val L1(MRI->CT): {val_loss:.4f}")

        if epoch > warm_up:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_G_w = copy.deepcopy(generator.state_dict())
                best_D_w = copy.deepcopy(discriminator.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping > 0 and epochs_no_improve >= early_stopping:
                    print(f"\nEarly Stopping! Epoch: {epoch}")
                    early_stop = True
        if early_stop:
            break

    print("Training completo.")
    print(f"Best epoch: {best_epoch}  Best val loss: {best_loss:.4f}")

    generator.load_state_dict(best_G_w)
    discriminator.load_state_dict(best_D_w)

    # salvataggio checkpoint
    save_checkpoint(generator, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G"]))
    save_checkpoint(discriminator, opt_D, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D"]))

    history = pd.DataFrame.from_dict(history, orient="index").transpose()
    return generator, discriminator, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(history["train_loss_G"], label="Train G")
    plt.plot(history["train_loss_D"], label="Train D")
    plt.plot(history["val_loss"], label="Val L1 (MRI->CT)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


def test_4metrics_stargan(
    generator,
    loader_test,
    device,
    outputs_dir,
    save_output: bool = True,
):

    generator.eval()

    mse_metric = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    loop_test = tqdm(loader_test, leave=True)
    running_mse = running_psnr = running_vif = running_ssim = 0.0

    with torch.no_grad():
        for mri, ct, file_names in loop_test:
            mri = mri.to(device)
            ct = ct.to(device)

            c_mri = torch.tensor([[1.0, 0.0]], device=device).repeat(mri.size(0), 1)
            c_ct  = torch.tensor([[0.0, 1.0]], device=device).repeat(mri.size(0), 1)

            with torch.cuda.amp.autocast():
                fake_ct = generator(mri, c_ct)  # in [-1,1]

                if save_output:
                    for out_img, fname in zip(fake_ct, file_names):
                        filename_output = f"{fname}_stargan_ct.png"
                        save_image(out_img, os.path.join(outputs_dir, filename_output))

                # porto in [0,1] per metriche
                fake_ct_01 = (fake_ct.float() + 1.0) / 2.0
                ct_01      = (ct.float()      + 1.0) / 2.0

                mse  = mse_metric(fake_ct_01, ct_01)
                psnr = psnr_metric(fake_ct_01, ct_01)
                vif  = vif_p(fake_ct_01, ct_01, data_range=1.0)
                ssim = structural_similarity_index_measure(fake_ct_01, ct_01, data_range=1.0)

            running_mse   += mse.item()  * mri.size(0)
            running_psnr  += psnr.item() * mri.size(0)
            running_vif   += vif.item()  * mri.size(0)
            running_ssim  += ssim.item() * mri.size(0)

    n = len(loader_test.dataset)
    return (
        running_mse / n,
        running_psnr / n,
        running_vif / n,
        running_ssim / n,
    )
@torch.no_grad()
def test_stargan(generator, loader_test, mse_metric, psnr_metric, device,
                 c_trg_fixed,
                 outputs_dir=None, save_outputs=False, *,
                 amp=True, lpips_backbone="alex", is_splits=10,
                 return_per_image=False):

    generator.eval()

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


        c_trg = c_trg_fixed.view(1, -1).repeat(bsz, 1)

        with scaler_ctx():
            fake_x2 = generator(x1, c_trg)

        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img.clamp(0,1), os.path.join(outputs_dir, f"{name}_output.png"))


        mse_val  = _to_scalar(mse_metric(fake_x2, x2))
        psnr_val = _to_scalar(psnr_metric(fake_x2, x2))


        with torch.cuda.amp.autocast(enabled=False):
            fake_clamped = fake_x2.clamp(0.0, 1.0).float()
            real_clamped = x2.clamp(0.0, 1.0).float()

            vif_val  = _to_scalar(piq_vif(fake_clamped, real_clamped))
            ssim_val = _to_scalar(tm_ssim(fake_clamped, real_clamped, data_range=1.0))

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

        # Inception (IS + FID)
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