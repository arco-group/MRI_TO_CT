import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure

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

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    z = mu + eps * sigma
    """
    if mu.is_cuda:
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    std = torch.exp(logvar / 2)
    eps = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
    return eps * std + mu


def train_fn(
    generator,
    encoder,
    D_VAE,
    D_LR,
    loader,
    opt_E,
    opt_G,
    opt_D_VAE,
    opt_D_LR,
    l1_pix,
    lambdas,
    device,
    g_scaler: torch.cuda.amp.GradScaler,
    d_vae_scaler: torch.cuda.amp.GradScaler,
    d_lr_scaler: torch.cuda.amp.GradScaler,
):

    generator.train()
    encoder.train()
    D_VAE.train()
    D_LR.train()

    loop = tqdm(loader, leave=True)

    lambda_pixel = lambdas["lambda_pixel"]
    lambda_latent = lambdas["lambda_latent"]
    lambda_kl = lambdas["lambda_kl"]

    for idx, (img_A, img_B, ids) in enumerate(loop):
        img_A = img_A.to(device)
        img_B = img_B.to(device)

        batch_size = img_A.size(0)
        if img_A.is_cuda:
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor

        # -------------------------------
        #  Train Generator + Encoder
        # -------------------------------
        opt_E.zero_grad(set_to_none=True)
        opt_G.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            # cVAE branch
            mu, logvar = encoder(img_B)
            z_enc = reparameterize(mu, logvar)
            fake_B_vae = generator(img_A, z_enc)

            # Pixel loss
            loss_pixel = l1_pix(fake_B_vae, img_B)

            # KL loss
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            loss_kl = loss_kl / batch_size

            # Adversarial cVAE-GAN
            loss_VAE_GAN = D_VAE.compute_loss(fake_B_vae, is_real=True)

            # cLR branch
            z_rand = Variable(Tensor(np.random.normal(0, 1, (batch_size, mu.shape[1]))))
            fake_B_lr = generator(img_A, z_rand)
            loss_LR_GAN = D_LR.compute_loss(fake_B_lr, is_real=True)

            # Loss totale (G+E)
            loss_GE = loss_VAE_GAN + loss_LR_GAN + lambda_pixel * loss_pixel + lambda_kl * loss_kl

        g_scaler.scale(loss_GE).backward(retain_graph=True)
        g_scaler.step(opt_E)
        g_scaler.step(opt_G)
        g_scaler.update()

        # ---------------------
        # Latent L1 loss (solo G)
        # ---------------------
        opt_G.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            mu_rec, _ = encoder(fake_B_lr)
            loss_latent = lambda_latent * l1_pix(mu_rec, z_rand)

        g_scaler.scale(loss_latent).backward()
        g_scaler.step(opt_G)
        g_scaler.update()

        # -----------------------------
        #  Train Discriminator VAE
        # -----------------------------
        opt_D_VAE.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss_D_VAE = D_VAE.compute_loss(img_B, is_real=True) + \
                         D_VAE.compute_loss(fake_B_vae.detach(), is_real=False)
        d_vae_scaler.scale(loss_D_VAE).backward()
        d_vae_scaler.step(opt_D_VAE)
        d_vae_scaler.update()

        # -----------------------------
        #  Train Discriminator LR
        # -----------------------------
        opt_D_LR.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss_D_LR = D_LR.compute_loss(img_B, is_real=True) + \
                        D_LR.compute_loss(fake_B_lr.detach(), is_real=False)
        d_lr_scaler.scale(loss_D_LR).backward()
        d_lr_scaler.step(opt_D_LR)
        d_lr_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                loss_GE=loss_GE.item(),
                loss_latent=loss_latent.item(),
                loss_D_VAE=loss_D_VAE.item(),
                loss_D_LR=loss_D_LR.item(),
            )
    return generator, encoder, D_VAE, D_LR

def eval_fn(generator, encoder, loader, criterion, device, outputs_dir):

    generator.eval()
    encoder.eval()

    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for idx, (img_A, img_B, file_names) in enumerate(loop):
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            with torch.cuda.amp.autocast():
                mu, logvar = encoder(img_B)
                z = reparameterize(mu, logvar)
                fake_B = generator(img_A, z)

                if outputs_dir:
                    for out_img, fn in zip(fake_B, file_names):
                        filename_output = f"{fn}_output.png"
                        save_image(out_img, os.path.join(outputs_dir, filename_output))

                loss = criterion(fake_B, img_B)

            running_loss += loss.item() * img_A.size(0)
            n_samples += img_A.size(0)

    epoch_loss = running_loss / max(n_samples, 1)
    return epoch_loss


def train_bicyclegan(
    generator,
    encoder,
    D_VAE,
    D_LR,
    data_loaders,
    opt_E,
    opt_G,
    opt_D_VAE,
    opt_D_LR,
    l1_pix,
    lambdas,
    early_stopping_criterion,
    model_fold_dir,
    cfg_trainer,
    device,
):
    g_scaler = torch.cuda.amp.GradScaler()
    d_vae_scaler = torch.cuda.amp.GradScaler()
    d_lr_scaler = torch.cuda.amp.GradScaler()

    best_gen_wts = copy.deepcopy(generator.state_dict())
    best_enc_wts = copy.deepcopy(encoder.state_dict())
    best_disc_vae_wts = copy.deepcopy(D_VAE.state_dict())
    best_disc_lr_wts = copy.deepcopy(D_LR.state_dict())
    best_loss = np.Inf
    best_epoch = -1

    history = {"train_loss": [], "val_loss": []}

    epochs_no_improve = 0
    early_stop = False
    numero_di_epoche = 0

    for epoch in range(cfg_trainer["max_epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg_trainer['max_epochs']}")
        # TRAIN
        generator, encoder, D_VAE, D_LR = train_fn(
            generator,
            encoder,
            D_VAE,
            D_LR,
            data_loaders["train"],
            opt_E,
            opt_G,
            opt_D_VAE,
            opt_D_LR,
            l1_pix,
            lambdas,
            device,
            g_scaler,
            d_vae_scaler,
            d_lr_scaler,
        )

        # VAL
        val_loss = eval_fn(
            generator=generator,
            encoder=encoder,
            loader=data_loaders["val"],
            criterion=early_stopping_criterion,
            device=device,
            outputs_dir=False,
        )
        train_loss = eval_fn(
            generator=generator,
            encoder=encoder,
            loader=data_loaders["train"],
            criterion=early_stopping_criterion,
            device=device,
            outputs_dir=False,
        )

        history["val_loss"].append(val_loss)
        history["train_loss"].append(train_loss)

        print(f"Train L1: {train_loss:.6f}  |  Val L1: {val_loss:.6f}")

        #Early stopping dopo warmup
        if epoch > cfg_trainer["warm_up"]:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_gen_wts = copy.deepcopy(generator.state_dict())
                best_enc_wts = copy.deepcopy(encoder.state_dict())
                best_disc_vae_wts = copy.deepcopy(D_VAE.state_dict())
                best_disc_lr_wts = copy.deepcopy(D_LR.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg_trainer["early_stopping"]:
                    print(f"\nEarly Stopping! Total epochs: {epoch+1}")
                    early_stop = True
            if early_stop:
                break

    print("Training e validation complete")
    if "best_epoch" in locals():
        print("Best epoch:", best_epoch)
    print("Best val Loss: {:.6f}".format(best_loss))
    if numero_di_epoche%1 == 0:
            print('numero di epoche finite: '+str(numero_di_epoche))



    generator.load_state_dict(best_gen_wts)
    encoder.load_state_dict(best_enc_wts)
    D_VAE.load_state_dict(best_disc_vae_wts)
    D_LR.load_state_dict(best_disc_lr_wts)


    save_checkpoint(
        generator, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN"])
    )
    save_checkpoint(
        encoder, opt_E, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_ENC"])
    )
    save_checkpoint(
        D_VAE, opt_D_VAE, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DISC_VAE"])
    )
    save_checkpoint(
        D_LR, opt_D_LR, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DISC_LR"])
    )

    history = pd.DataFrame.from_dict(history, orient="index").transpose()
    return generator, encoder, D_VAE, D_LR, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss (fake_B vs B)")
    plt.title("Training and Validation Losses - BicycleGAN")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


def test_4metrics(
    generator,
    encoder,
    loader,
    mse_metric,
    psnr_metric,
    device,
    outputs_dir,
    save_output: bool,
):

    generator.eval()
    encoder.eval()

    loop = tqdm(loader, leave=True)
    running_mse = 0.0
    running_psnr = 0.0
    running_vif = 0.0
    running_ssim = 0.0
    n_samples = 0

    with torch.no_grad():
        for idx, (img_A, img_B, file_names) in enumerate(loop):
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            with torch.cuda.amp.autocast():
                mu, logvar = encoder(img_B)
                z = reparameterize(mu, logvar)
                fake_B = generator(img_A, z)

                if save_output:
                    for out_img, fn in zip(fake_B, file_names):
                        filename_output = f"{fn}_output.png"
                        save_image(out_img, os.path.join(outputs_dir, filename_output))

                mse = mse_metric(fake_B, img_B)
                psnr = psnr_metric(fake_B, img_B)

                # VIF & SSIM richiedono float32
                fake_32 = fake_B.float()
                real_32 = img_B.float()

                vif = vif_p(fake_32, real_32)
                ssim = structural_similarity_index_measure(
                    fake_32, real_32, data_range=1.0
                )

            bs = img_A.size(0)
            running_mse += mse.item() * bs
            running_psnr += psnr.item() * bs
            running_vif += vif.item() * bs
            running_ssim += ssim.item() * bs
            n_samples += bs

    epoch_mse = running_mse / max(n_samples, 1)
    epoch_psnr = running_psnr / max(n_samples, 1)
    epoch_vif = running_vif / max(n_samples, 1)
    epoch_ssim = running_ssim / max(n_samples, 1)

    return epoch_mse, epoch_psnr, epoch_vif, epoch_ssim
def test_bicyclegan(generator, encoder, loader_test, mse_metric, psnr_metric, device,
                    outputs_dir=None, save_outputs=False, *,
                    amp=True, lpips_backbone="alex", is_splits=10,
                    return_per_image=False):

    generator.eval()
    encoder.eval()

    metric_sums = {
        "mse": 0.0,
        "psnr": 0.0,
        "vif": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }
    per_image = {
        "mse": [],
        "psnr": [],
        "vif": [],
        "ssim": [],
        "lpips": []
    }

    n_images = 0

    # LPIPS
    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    # Inception per IS/FID
    inc_cls = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat.fc = nn.Identity()

    logits_all = []
    feats_real_all = []
    feats_fake_all = []

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for batch_idx, batch in enumerate(pbar):
        x1, x2, file_names = batch  # ImgDataset: (LE, REC, name)
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)

        bsz = x1.size(0)
        n_images += bsz

        # ---- forward BicycleGAN ----
        # encoder produce z da x2 (target reale), generator usa x1+z
        with torch.no_grad():
            mu, logvar = encoder(x2)

            z = mu

        with scaler_ctx():
            fake_x2 = generator(x1, z)

        # save outputs
        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img.clamp(0, 1),
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

        # ---- Inception (FID/IS) ----
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