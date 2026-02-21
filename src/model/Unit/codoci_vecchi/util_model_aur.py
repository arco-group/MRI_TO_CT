
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
from piq import ms_ssim as piq_ms_ssim, vif_p as piq_vif
import lpips
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR


# ------------------------------------------------------------------------------
# Loss helpers (MSE-GAN targets)
# ------------------------------------------------------------------------------
def _real_like(x):  # target=1
    return torch.ones_like(x)

def _fake_like(x):  # target=0
    return torch.zeros_like(x)


# ------------------------------------------------------------------------------
# TRAIN ONE EPOCH
# ------------------------------------------------------------------------------
def train_fn(
    E1, E2, G1, G2, D1, D2,
    loader,
    opt_G, opt_D1, opt_D2,
    mse_gan, l1_pix,
    lambdas, device,
    g_scaler: torch.cuda.amp.GradScaler,
    d1_scaler: torch.cuda.amp.GradScaler,
    d2_scaler: torch.cuda.amp.GradScaler,
):
    """
    Convenzione domini:
      X1 = MRI (sorgente)  -> enc E1 -> z1 -> G2 -> CT (bersaglio)
      X2 = CT  (bersaglio) -> enc E2 -> z2 -> G1 -> MRI
    D1 discrimina su dominio MRI, D2 su dominio CT.
    """
    E1.train(); E2.train(); G1.train(); G2.train(); D1.train(); D2.train()

    loop = tqdm(loader, leave=True)

    # --- accumulatore per train L1(MRI->CT) ---
    running_l1_mri2ct = 0.0
    n_samples = 0

    for _, (x1, x2, _) in enumerate(loop):
        x1 = x1.to(device, non_blocking=True).float()  # MRI
        x2 = x2.to(device, non_blocking=True).float()  # PCCT

        # ---------------------
        #    Train Generators
        # ---------------------
        opt_G.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            mu1, z1 = E1(x1)
            mu2, z2 = E2(x2)

            # ricostruzioni
            recon_x1 = G1(z1)
            recon_x2 = G2(z2)

            # traduzioni
            fake_x2 = G2(z1)  # MRI->CT
            fake_x1 = G1(z2)  # CT->MRI

            # cycle translation
            mu1_c, z1_c = E1(fake_x1)
            mu2_c, z2_c = E2(fake_x2)
            cyc_x1 = G1(z2_c)
            cyc_x2 = G2(z1_c)

            # avversarie (MSE-GAN)
            pred_fake_on_MRI = D1(fake_x1)
            pred_fake_on_CT  = D2(fake_x2)
            loss_gan_1 = mse_gan(pred_fake_on_MRI, torch.ones_like(pred_fake_on_MRI))
            loss_gan_2 = mse_gan(pred_fake_on_CT,  torch.ones_like(pred_fake_on_CT))

            # KL (sui mu)
            loss_kl_1  = torch.mean(mu1 ** 2)
            loss_kl_2  = torch.mean(mu2 ** 2)

            # KL su mu dopo traduzione
            loss_kl_1_ = torch.mean(mu1_c ** 2)
            loss_kl_2_ = torch.mean(mu2_c ** 2)

            # pixel-wise
            loss_id_1 = l1_pix(recon_x1, x1)   # identity (ricostruzione MRI)
            loss_id_2 = l1_pix(recon_x2, x2)   # identity (ricostruzione CT)
            loss_cyc_1 = l1_pix(cyc_x1, x1)    # ciclo MRI
            loss_cyc_2 = l1_pix(cyc_x2, x2)    # ciclo CT

            # --- L1 supervisionale su MRI->CT per MONITORAGGIO (non entra in LG) ---
            loss_l1_mri2ct_batch = l1_pix(fake_x2, x2)

            # totale Generators
            LG = (
                lambdas["lambda_0"] * (loss_gan_1 + loss_gan_2)
                + lambdas["lambda_1"] * (loss_kl_1 + loss_kl_2)
                + lambdas["lambda_2"] * (loss_id_1 + loss_id_2)
                + lambdas["lambda_3"] * (loss_kl_1_ + loss_kl_2_)
                + lambdas["lambda_4"] * (loss_cyc_1 + loss_cyc_2)
            )

        g_scaler.scale(LG).backward()
        g_scaler.step(opt_G)
        g_scaler.update()

        # ---------------------
        #   Train Discriminator 1 (MRI)
        # ---------------------
        opt_D1.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred_real_1 = D1(x1)
            pred_fake_1 = D1(fake_x1.detach())
            LD1 = (
                mse_gan(pred_real_1, torch.ones_like(pred_real_1)) +
                mse_gan(pred_fake_1, torch.zeros_like(pred_fake_1))
            )
        d1_scaler.scale(LD1).backward()
        d1_scaler.step(opt_D1)
        d1_scaler.update()

        # ---------------------
        #   Train Discriminator 2 (CT)
        # ---------------------
        opt_D2.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred_real_2 = D2(x2)
            pred_fake_2 = D2(fake_x2.detach())
            LD2 = (
                mse_gan(pred_real_2, torch.ones_like(pred_real_2)) +
                mse_gan(pred_fake_2, torch.zeros_like(pred_fake_2))
            )
        d2_scaler.scale(LD2).backward()
        d2_scaler.step(opt_D2)
        d2_scaler.update()

        # --- accumulo train L1(MRI->CT) medio per epoca ---
        bs = x1.size(0)
        running_l1_mri2ct += loss_l1_mri2ct_batch.item() * bs
        n_samples += bs

    avg_train_l1_mri2ct = running_l1_mri2ct / max(1, n_samples)

    # restituisco anche un dict di metriche di training
    train_metrics = {
        "train_l1_mri2ct": avg_train_l1_mri2ct
    }

    # opzionale: si può restituire le ultime perdite
    return E1, E2, G1, G2, D1, D2, train_metrics


# ------------------------------------------------------------------------------
# EVAL (usiamo loss L1 sulla mappa tradotta MRI->CT rispetto al CT target)
# ------------------------------------------------------------------------------
@torch.no_grad()
def eval_fn(E1, G2, loader, criterion, device, outputs_dir=None):
    E1.eval(); G2.eval()
    loop = tqdm(loader, leave=True)
    tot_loss = 0.0
    n = 0
    for x1, x2, file_names in loop:
        x1 = x1.to(device).float()  # MRI
        x2 = x2.to(device).float()  # CT
        with torch.cuda.amp.autocast():
            _, z1 = E1(x1)
            fake_x2 = G2(z1)  # MRI->CT
        loss = criterion(fake_x2, x2)
        tot_loss += loss.item() * x1.size(0)
        n += x1.size(0)

        if outputs_dir:
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img, os.path.join(outputs_dir, f"{name}_output.png"))
    return tot_loss / max(1, n)


# ------------------------------------------------------------------------------
# LOOP di training completo con Early Stopping + salvataggi
# ------------------------------------------------------------------------------
def train_unit(
    E1, E2, G1, G2, D1, D2,
    data_loaders, early_stopping_criterion,
    opt_G, opt_D1, opt_D2, l1_pix, mse_gan, model_fold_dir,
    cfg_trainer, device
):
    # lambdas dalla cfg (coerenti con UNIT GitHub)
    lambdas = dict(
        lambda_0 = cfg_trainer.get("lambda_0", 10.0),
        lambda_1 = cfg_trainer.get("lambda_1", 0.1),
        lambda_2 = cfg_trainer.get("lambda_2", 100.0),
        lambda_3 = cfg_trainer.get("lambda_3", 0.1),
        lambda_4 = cfg_trainer.get("lambda_4", 100.0),
    )

    g_scaler  = torch.cuda.amp.GradScaler()
    d1_scaler = torch.cuda.amp.GradScaler()
    d2_scaler = torch.cuda.amp.GradScaler()

    best_wts = {
        "E1": copy.deepcopy(E1.state_dict()),
        "E2": copy.deepcopy(E2.state_dict()),
        "G1": copy.deepcopy(G1.state_dict()),
        "G2": copy.deepcopy(G2.state_dict()),
        "D1": copy.deepcopy(D1.state_dict()),
        "D2": copy.deepcopy(D2.state_dict()),
    }

    best_loss = np.Inf
    best_epoch = -1
    hist = {"train_loss": [], "val_loss": []}
    epochs_no_improve = 0
    early_stop = False

    max_epochs   = cfg_trainer["max_epochs"]
    warm_up      = cfg_trainer["warm_up"]
    patience     = cfg_trainer["early_stopping"]

    # -------------------------
    # SCHEDULER LR (tipo CycleGAN)
    # -------------------------
    # se non specificato, inizio decay a metà training
    decay_start_epoch = cfg_trainer.get("decay_start_epoch", max_epochs // 2)

    def lambda_lr(epoch):
        if epoch < decay_start_epoch:
            return 1.0
        else:
            # decay lineare fino a 0 tra decay_start_epoch e max_epochs
            return max(0.0, 1.0 - float(epoch - decay_start_epoch) / float(max_epochs - decay_start_epoch))

    scheduler_G = LambdaLR(opt_G, lr_lambda=lambda_lr)
    scheduler_D1 = LambdaLR(opt_D1, lr_lambda=lambda_lr)
    scheduler_D2 = LambdaLR(opt_D2, lr_lambda=lambda_lr)

    for epoch in range(max_epochs):
        # TRAIN
        E1, E2, G1, G2, D1, D2, train_metrics = train_fn(
            E1, E2, G1, G2, D1, D2,
            data_loaders["train"],
            opt_G, opt_D1, opt_D2,
            mse_gan, l1_pix,
            lambdas, device,
            g_scaler, d1_scaler, d2_scaler
        )

        train_loss = train_metrics["train_l1_mri2ct"]

        # EVAL su val (L1 su MRI->CT)
        val_loss  = eval_fn(E1, G2, data_loaders["val"], early_stopping_criterion, device, outputs_dir=None)
        #train_loss = eval_fn(E1, G2, data_loaders["train"], early_stopping_criterion, device, outputs_dir=None)

        hist["val_loss"].append(val_loss)
        hist["train_loss"].append(train_loss)

        # Early stopping dopo warmup
        if epoch > warm_up:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_wts = {
                    "E1": copy.deepcopy(E1.state_dict()),
                    "E2": copy.deepcopy(E2.state_dict()),
                    "G1": copy.deepcopy(G1.state_dict()),
                    "G2": copy.deepcopy(G2.state_dict()),
                    "D1": copy.deepcopy(D1.state_dict()),
                    "D2": copy.deepcopy(D2.state_dict()),
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly Stopping @ epoch {epoch} (best={best_epoch}, val={best_loss:.6f})")
                    early_stop = True
        print(f"[Epoch {epoch+1}/{max_epochs}] train_L1(MRI->PCCT)={train_loss:.6f} | val_L1(MRI->PCCT)={val_loss:.6f}")

        # STEP scheduler a fine epoca
        scheduler_G.step()
        scheduler_D1.step()
        scheduler_D2.step()

        if early_stop:
            break

    # ripristina best pesi
    E1.load_state_dict(best_wts["E1"])
    E2.load_state_dict(best_wts["E2"])
    G1.load_state_dict(best_wts["G1"])
    G2.load_state_dict(best_wts["G2"])
    D1.load_state_dict(best_wts["D1"])
    D2.load_state_dict(best_wts["D2"])

    # Save model
    save_checkpoint(E1, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_E1"]))
    save_checkpoint(E2, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_E2"]))
    save_checkpoint(G1, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G1"]))
    save_checkpoint(G2, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G2"]))
    save_checkpoint(D1, opt_D1, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D1"]))
    save_checkpoint(D2, opt_D2, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D2"]))

    history = pd.DataFrame.from_dict(hist, orient="index").transpose()

    return E1, E2, G1, G2, D1, D2, history


# ------------------------------------------------------------------------------
# Plot training (Loss)
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# TEST metrics (MRI->CT): MSE / PSNR / VIF / SSIM + salvataggi PNG
# ------------------------------------------------------------------------------
@torch.no_grad()
def test_4metrics_unit(
    E1, G2,
    loader,
    mse,
    psnr,
    device,
    outputs_dir=None,
    save_output=True
):
    E1.eval(); G2.eval()

    run_mse = run_psnr = run_vif = run_ssim = 0.0
    n = 0

    loop = tqdm(loader, leave=True)
    with torch.no_grad():
        for x1, x2, file_names in loop:  # MRI, CT
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()

            with torch.cuda.amp.autocast():
                _, z1 = E1(x1)
                fake_x2 = G2(z1)

            # salvataggio
            if save_output and outputs_dir:
                for out_img, name in zip(fake_x2, file_names):
                    save_image(out_img, os.path.join(outputs_dir, f"{name}_output.png"))

            # metriche
            v_mse  = mse(fake_x2, x2).item()
            v_psnr = psnr(fake_x2, x2).item()
            v_vif  = vif_p(fake_x2.float(), x2.float()).item()
            v_ssim = ssim_fn(fake_x2.float(), x2.float(), data_range=1.0).item()

            bsz = x1.size(0)
            run_mse  += v_mse  * bsz
            run_psnr += v_psnr * bsz
            run_vif  += v_vif  * bsz
            run_ssim += v_ssim * bsz
            n += bsz

    return (run_mse / n, run_psnr / n, run_vif / n, run_ssim / n)



def test_unit(
    E1,
    G2,
    loader_test,
    mse_metric,
    psnr_metric,
    device,
    outputs_dir=None,
    save_outputs=False,
    *,
    amp=True,
    lpips_backbone="alex",
    compute_is=True,
    compute_fid=True,
    is_splits=10,
):
    """
    Valuta un modello di image-to-image translation su più metriche.

    Assunzioni:
      - Le immagini in `loader_test` sono in [0, 1] (float) con shape [B, C, H, W].
      - Il dataloader restituisce tuple (lowenergy, recombined, file_names).
      - `mse_metric` e `psnr_metric` sono istanze torchmetrics o funzioni compatibili (batch-wise).

    Parametri opzionali:
      - amp: usa autocast per lo step di forward del generatore.
      - compute_is/fid: calcola Inception Score e/o FID.
      - save_outputs/outputs_dir: salva le uscite PNG clampate in [0,1].
      - lpips_backbone: "alex" | "vgg" | "squeeze".
      - is_splits: numero di split per Inception Score.

    Ritorna:
      dict con medie su tutto il dataset: mse, psnr, vif, ssim, lpips, ms_ssim, (fid), (is_mean, is_std)
    """

    E1.eval(); G2.eval()

    # --- accumulators (somme pesate per numero di immagini) ---
    sums = {
        "mse": 0.0,
        "psnr": 0.0,
        "vif": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "ms_ssim": 0.0,
    }
    n_images = 0

    # --- metric models ---
    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    # Inception per IS/FID (caricata solo se serve)
    inc_cls = inc_feat = None
    if compute_is or compute_fid:
        inc_cls = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
        inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
        inc_feat.fc = nn.Identity()  # features 2048

    logits_all = []     # per IS (logits su fake)
    feats_real_all = [] # per FID (features 2048 su real)
    feats_fake_all = [] # per FID (features 2048 su fake)

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for idx, batch in enumerate(pbar):
        # --- estrazione batch ---
        x1, x2, file_names = batch
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)
        bsz = x1.size(0)
        n_images += bsz

        # --- forward generatore ---
        with scaler_ctx():
            _, z1 = E1(x1)
            fake_x2 = G2(z1)

        # --- opzionale: salvataggio PNG in [0,1] ---
        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            # salva ogni immagine del batch
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img.clamp(0, 1), os.path.join(outputs_dir, f"{name}_output.png"))

        # --- metriche base (tensori in [0,1], float32) ---
        mse_val  = _to_scalar(mse_metric(fake_x2, x2))
        psnr_val = _to_scalar(psnr_metric(fake_x2, x2))

        # piq.vif_p richiede stesso dtype dei pesi -> usiamo float32 coerente
        #vif_val  = _to_scalar(piq_vif(fake_x2, x2))  # in [0, +)
        with torch.cuda.amp.autocast(enabled=False):
            vif_val = _to_scalar(piq_vif(fake_x2.float(), x2.float()))

        # torchmetrics SSIM: data_range=1.0, mediamo sul batch
        ssim_val = _to_scalar(tm_ssim(fake_x2, x2, data_range=1.0))

        # --- LPIPS (richiede tensori in [-1,1] e 3 canali) ---
        a_lp = _prep_for_lpips(fake_x2)
        b_lp = _prep_for_lpips(x2)
        lpips_val = _to_scalar(lpips_fn(a_lp, b_lp).mean())

        # --- MS-SSIM (piq) su [0,1], 3 canali per robustezza ---
        a_3c = _ensure_3ch(fake_x2)
        b_3c = _ensure_3ch(x2)
        #ms_val = _to_scalar(piq_ms_ssim(a_3c, b_3c, data_range=1.0, reduction="mean"))

        # --- accumula somme pesate ---
        sums["mse"] += mse_val * bsz
        sums["psnr"] += psnr_val * bsz
        sums["vif"] += vif_val * bsz
        sums["ssim"] += ssim_val * bsz
        sums["lpips"] += lpips_val * bsz
        #sums["ms_ssim"] += ms_val * bsz

        # --- Inception (IS/FID) ---
        if compute_is or compute_fid:

            with torch.no_grad():  # sempre no_grad per IS/FID
                with torch.cuda.amp.autocast(enabled=False):

                    fake_inc = _prep_for_inception(fake_x2).float().to(device)
                    real_inc = _prep_for_inception(x2).float().to(device)

                    if compute_is:
                        logits = inc_cls(fake_inc)
                        logits_all.append(logits.detach().cpu())

                    if compute_fid:
                        f_fake = inc_feat(fake_inc)
                        f_real = inc_feat(real_inc)

                        feats_fake_all.append(f_fake.detach().cpu().numpy())
                        feats_real_all.append(f_real.detach().cpu().numpy())

    # --- medie finali (su tutte le immagini, non per numero di batch) ---
    eps = 1e-12
    results = {k: (v / max(n_images, 1)) for k, v in sums.items()}

    # --- FID ---
    if compute_fid and len(feats_fake_all) > 0 and len(feats_real_all) > 0:
        Ffake = np.concatenate(feats_fake_all, axis=0)  # [N,2048]
        Freal = np.concatenate(feats_real_all, axis=0)
        mu_f, sig_f = Ffake.mean(axis=0), np.cov(Ffake, rowvar=False)
        mu_r, sig_r = Freal.mean(axis=0), np.cov(Freal, rowvar=False)
        results["fid"] = _frechet_distance(mu_r, sig_r, mu_f, sig_f)

    # --- Inception Score ---
    if compute_is and len(logits_all) > 0:
        logits_cat = torch.cat(logits_all, dim=0)       # [N,1000] on CPU
        is_mean, is_std = _inception_score_from_logits(logits_cat, splits=is_splits)
        results["is_mean"] = float(is_mean)
        results["is_std"]  = float(is_std)

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
    # x: [B,C,H,W] con C in {1,3}
    if x.size(1) == 3:
        return x
    return x.repeat(1, 3, 1, 1)

def _prep_for_lpips(x: torch.Tensor) -> torch.Tensor:
    # LPIPS vuole tensori in [-1,1] e 3 canali
    x = x.clamp(0, 1)
    x = _ensure_3ch(x)
    x = x * 2.0 - 1.0
    return x

def _prep_for_inception(x: torch.Tensor) -> torch.Tensor:
    # Resize a 299 e normalizzazione ImageNet
    # Evita dipendenze esterne: usa F.interpolate + normalizzazione manuale
    x = _ensure_3ch(x.clamp(0, 1))
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    # Normalizzazione IMAGENET
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std  = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (x - mean) / std
    return x

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID classico: d^2 = ||mu1-mu2||^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma1*Sigma2)^{1/2})
    """
    from scipy import linalg
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # sqrtm può essere instabile; aggiungi eps alla diagonale se necessario
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # eventuale parte immaginaria numerica
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)

def _inception_score_from_logits(logits: torch.Tensor, splits: int = 10):
    """
    Calcola IS dai logits (non softmax) di Inception v3.
    Ritorna (IS_mean, IS_std).
    """
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
