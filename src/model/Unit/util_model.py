import sys; print('Python %s on %s' % (sys.version, sys.platform))
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
# ------------------------------------------------------------------------------
# Loss helpers (MSE-GAN targets)
# ------------------------------------------------------------------------------
#def _real_like(x):  # target=1
    #return torch.ones_like(x)

#def _fake_like(x):  # target=0
    #return torch.zeros_like(x)


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

    E1.train(); E2.train(); G1.train(); G2.train(); D1.train(); D2.train()
    loop = tqdm(loader, leave=True)

    for _, (x1, x2, _) in enumerate(loop):

        x1 = x1.to(device, non_blocking=True).float()  # MRI
        x2 = x2.to(device, non_blocking=True).float()  # CT


        opt_G.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            mu1, z1 = E1(x1)
            mu2, z2 = E2(x2)


            recon_x1 = G1(z1)
            recon_x2 = G2(z2)


            fake_x2 = G2(z1)  # MRI->CT
            fake_x1 = G1(z2)  # CT->MRI


            mu1_c, z1_c = E1(fake_x1)
            mu2_c, z2_c = E2(fake_x2)
            cyc_x1 = G1(z2_c)
            cyc_x2 = G2(z1_c)


            pred_fake_on_MRI = D1(fake_x1)
            pred_fake_on_CT  = D2(fake_x2)
            loss_gan_1 = mse_gan(pred_fake_on_MRI, torch.ones_like(pred_fake_on_MRI))
            loss_gan_2 = mse_gan(pred_fake_on_CT,  torch.ones_like(pred_fake_on_CT))


            loss_kl_1  = torch.mean(mu1 ** 2)
            loss_kl_2  = torch.mean(mu2 ** 2)


            loss_kl_1_ = torch.mean(mu1_c ** 2)
            loss_kl_2_ = torch.mean(mu2_c ** 2)

            # pixel-wise
            loss_id_1 = l1_pix(recon_x1, x1)
            loss_id_2 = l1_pix(recon_x2, x2)
            loss_cyc_1 = l1_pix(cyc_x1, x1)
            loss_cyc_2 = l1_pix(cyc_x2, x2)

            #  Generators
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

       #train discriminator1
        opt_D1.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred_real_1 = D1(x1)
            pred_fake_1 = D1(fake_x1.detach())
            LD1 = (
                mse_gan(D1(x1), torch.ones_like(pred_real_1)) +
                mse_gan(D1(fake_x1.detach()), torch.zeros_like(pred_fake_1))
            )
        d1_scaler.scale(LD1).backward()
        d1_scaler.step(opt_D1)
        d1_scaler.update()

        #train discriminator 2
        opt_D2.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred_real_2 = D2(x2)
            pred_fake_2 = D2(fake_x2.detach())
            LD2 = 0.5 * (
                mse_gan(pred_real_2, torch.ones_like(pred_real_2)) +
                mse_gan(pred_fake_2, torch.zeros_like(pred_fake_2))
            )
        d2_scaler.scale(LD2).backward()
        d2_scaler.step(opt_D2)
        d2_scaler.update()

    return E1, E2, G1, G2, D1, D2



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



def train_unit(
    E1, E2, G1, G2, D1, D2,
    data_loaders, early_stopping_criterion,
    opt_G, opt_D1, opt_D2, l1_pix, mse_gan, model_fold_dir,
    cfg_trainer, device
):



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


    max_epochs   = cfg_trainer["max_epochs"]
    warm_up      = cfg_trainer["warm_up"]
    patience     = cfg_trainer["early_stopping"]

    # -----------------------------
    # MAIN CHECKPOINT PATH
    # -----------------------------
    main_ckpt_name = cfg_trainer.get("CHECKPOINT_MAIN", "UNIT_main.pth.tar")
    resume_path = os.path.join(model_fold_dir, main_ckpt_name)

    # -----------------------------
    # INIT / RESUME TRAINING STATE
    # -----------------------------
    start_epoch       = 0
    best_loss         = np.Inf
    best_epoch        = -1
    epochs_no_improve = 0
    hist = {"train_loss": [], "val_loss": []}

    if os.path.exists(resume_path):
        print(f"=> Trovato checkpoint MAIN UNIT: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        training_finished = ckpt.get("training_finished", False)


        E1.load_state_dict(ckpt["E1"])
        E2.load_state_dict(ckpt["E2"])
        G1.load_state_dict(ckpt["G1"])
        G2.load_state_dict(ckpt["G2"])
        D1.load_state_dict(ckpt["D1"])
        D2.load_state_dict(ckpt["D2"])

        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D1.load_state_dict(ckpt["opt_D1"])
        opt_D2.load_state_dict(ckpt["opt_D2"])

        hist              = ckpt.get("hist", hist)
        best_loss         = ckpt.get("best_loss", np.Inf)
        best_epoch        = ckpt.get("best_epoch", -1)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        start_epoch       = ckpt.get("epoch", 0)

        if training_finished:

            print("=> Allenamento già terminato in una run precedente. Salto il training.")

            history = pd.DataFrame.from_dict(hist, orient="index").transpose()
            return E1, E2, G1, G2, D1, D2, history

        print(f"   Riprendo da epoch {start_epoch}/{max_epochs}, "
              f"best_epoch={best_epoch}, best_loss={best_loss:.6f}, "
              f"epochs_no_improve={epochs_no_improve}")
    else:
        print("=> Nessun MAIN checkpoint UNIT trovato, parto da epoch 0")


    best_wts = {
        "E1": copy.deepcopy(E1.state_dict()),
        "E2": copy.deepcopy(E2.state_dict()),
        "G1": copy.deepcopy(G1.state_dict()),
        "G2": copy.deepcopy(G2.state_dict()),
        "D1": copy.deepcopy(D1.state_dict()),
        "D2": copy.deepcopy(D2.state_dict()),
    }

    early_stop = False


    for epoch in range(start_epoch, max_epochs):
        # ----------------- TRAIN -----------------
        E1, E2, G1, G2, D1, D2 = train_fn(
            E1, E2, G1, G2, D1, D2,
            data_loaders["train"],
            opt_G, opt_D1, opt_D2,
            mse_gan, l1_pix,
            lambdas, device,
            g_scaler, d1_scaler, d2_scaler
        )

        # ----------------- EVAL ------------------
        val_loss   = eval_fn(E1, G2, data_loaders["val"],   early_stopping_criterion, device, outputs_dir=None)
        train_loss = eval_fn(E1, G2, data_loaders["train"], early_stopping_criterion, device, outputs_dir=None)

        hist["val_loss"].append(val_loss)
        hist["train_loss"].append(train_loss)

        print(f"[Epoch {epoch+1}/{max_epochs}] "
              f"train_L1(MRI->CT)={train_loss:.6f} | val_L1(MRI->CT)={val_loss:.6f}")



        improved = False
        if val_loss < best_loss:
            improved = True
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


            E1.load_state_dict(best_wts["E1"])
            E2.load_state_dict(best_wts["E2"])
            G1.load_state_dict(best_wts["G1"])
            G2.load_state_dict(best_wts["G2"])
            D1.load_state_dict(best_wts["D1"])
            D2.load_state_dict(best_wts["D2"])


            save_checkpoint(E1, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_E1"]))
            save_checkpoint(E2, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_E2"]))
            save_checkpoint(G1, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G1"]))
            save_checkpoint(G2, opt_G, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_G2"]))
            save_checkpoint(D1, opt_D1, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D1"]))
            save_checkpoint(D2, opt_D2, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_D2"]))


        if epoch > warm_up:
            if not improved:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly Stopping @ epoch {epoch} "
                          f"(best={best_epoch}, val={best_loss:.6f})")
                    early_stop = True


        state = {
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_loss": float(best_loss),
            "epochs_no_improve": int(epochs_no_improve),
            "hist": hist,
            "training_finished": False,

            "E1": E1.state_dict(),
            "E2": E2.state_dict(),
            "G1": G1.state_dict(),
            "G2": G2.state_dict(),
            "D1": D1.state_dict(),
            "D2": D2.state_dict(),

            "opt_G":  opt_G.state_dict(),
            "opt_D1": opt_D1.state_dict(),
            "opt_D2": opt_D2.state_dict(),
        }
        torch.save(state, resume_path)

        if early_stop:
            break


    print('Training e validation complete')
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Loss: {:4f}'.format(best_loss))


    final_state = {
        "epoch": epoch + 1,
        "best_epoch": best_epoch,
        "best_loss": float(best_loss),
        "epochs_no_improve": int(epochs_no_improve),
        "hist": hist,
        "training_finished": True,

        "E1": E1.state_dict(),
        "E2": E2.state_dict(),
        "G1": G1.state_dict(),
        "G2": G2.state_dict(),
        "D1": D1.state_dict(),
        "D2": D2.state_dict(),

        "opt_G":  opt_G.state_dict(),
        "opt_D1": opt_D1.state_dict(),
        "opt_D2": opt_D2.state_dict(),
    }
    torch.save(final_state, resume_path)

    history = pd.DataFrame.from_dict(hist, orient="index").transpose()
    return E1, E2, G1, G2, D1, D2, history


#plot training
def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))

    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    #for c in ['train_loss','val_loss']:
        #plt.plot(history[c], label=c)
    #plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()


# ------------------------------------------------------------------------------
# TEST metrics (MRI->CT): MSE / PSNR / VIF / SSIM
# ------------------------------------------------------------------------------
@torch.no_grad()
def test_4metrics_unit(
    E1, G2,
    loader, mse, psnr,
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


            if save_output and outputs_dir:
                for out_img, name in zip(fake_x2, file_names):
                    save_image(out_img, os.path.join(outputs_dir, f"{name}_output.png"))

            # metrics
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
@torch.no_grad()
def test_unit(E1, G2, loader_test, mse_metric, psnr_metric, device,
              outputs_dir=None, save_outputs=False, *,
              amp=True, lpips_backbone="alex", is_splits=10,
              return_per_image=False):


    E1.eval()
    G2.eval()

    metric_sums = {"mse":0.0, "psnr":0.0, "vif":0.0, "ssim":0.0, "lpips":0.0}
    per_image = {k:[] for k in metric_sums.keys()}
    n_images = 0

    # LPIPS
    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    # Inception networks
    inc_cls  = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inc_feat.fc = nn.Identity()
    inc_feat.eval()

    logits_all = []
    feats_fake_all = []
    feats_real_all = []

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for batch_idx, batch in enumerate(pbar):
        x1, x2, file_names = batch
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)

        bsz = x1.size(0)
        n_images += bsz


        with scaler_ctx():
            _, z1 = E1(x1)    # encode MRI → shared space
            fake_x2 = G2(z1) # decode → CT


        # save output images
        if save_outputs and outputs_dir:
            for out_img, name in zip(fake_x2, file_names):
                save_image(out_img.clamp(0,1),
                           os.path.join(outputs_dir, f"{name}_output.png"))

        # base metrics
        mse_val  = _to_scalar(mse_metric(fake_x2, x2))
        psnr_val = _to_scalar(psnr_metric(fake_x2, x2))

        with torch.cuda.amp.autocast(enabled=False):
            vif_val  = _to_scalar(piq_vif(fake_x2.float(), x2.float()))
            ssim_val = _to_scalar(tm_ssim(fake_x2, x2, data_range=1.0))

        # LPIPS
        a_lp = _prep_for_lpips(fake_x2)
        b_lp = _prep_for_lpips(x2)
        lpips_val = _to_scalar(lpips_fn(a_lp, b_lp).mean())

        # accumulate
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

        # inception feats
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fake_inc = _prep_for_inception(fake_x2).float().to(device)
                real_inc = _prep_for_inception(x2).float().to(device)

                logits = inc_cls(fake_inc)
                logits_all.append(logits.detach().cpu())

                feats_fake_all.append(inc_feat(fake_inc).detach().cpu().numpy())
                feats_real_all.append(inc_feat(real_inc).detach().cpu().numpy())


    results = {}
    denom = max(n_images, 1)
    for k,v in metric_sums.items():
        results[k] = v / denom

    # FID
    if feats_fake_all:
        Ffake = np.concatenate(feats_fake_all,0)
        Freal = np.concatenate(feats_real_all,0)
        mu_f, sig_f = Ffake.mean(0), np.cov(Ffake, rowvar=False)
        mu_r, sig_r = Freal.mean(0), np.cov(Freal, rowvar=False)
        results["fid"] = _frechet_distance(mu_r, sig_r, mu_f, sig_f)
    else:
        results["fid"] = float("nan")

    # IS
    if logits_all:
        logits_cat = torch.cat(logits_all,0)
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