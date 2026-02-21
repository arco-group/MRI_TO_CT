import sys;

import torch.nn as nn

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
from piq import vif_p as piq_vif
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as tm_ssim

from torchvision.models.inception import inception_v3, Inception_V3_Weights
import lpips
import torch.nn.functional as F

import src.utils.util_general as util_general
from src.utils.util_general import save_checkpoint
from sys import platform


def train_fn(disc_REC, disc_LE, gen_LE, gen_REC, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, device, cfg_trainer):
    disc_REC.train()
    disc_LE.train()
    gen_LE.train()
    gen_REC.train()
    REC_reals = 0
    REC_fakes = 0
    loop = tqdm(loader, leave=True)
    for idx, (lowenergy, recombined, id) in enumerate(loop):  #
        lowenergy = lowenergy.to(device)
        recombined = recombined.to(device)
        # Train Discriminators
        with torch.cuda.amp.autocast():
            fake_recombined = gen_REC(lowenergy)
            D_REC_real = disc_REC(recombined)
            D_REC_fake = disc_REC(fake_recombined.detach())
            REC_reals += D_REC_real.mean().item()
            REC_fakes += D_REC_fake.mean().item()
            D_REC_real_loss = mse(D_REC_real, torch.ones_like(D_REC_real))  # real = 1
            D_REC_fake_loss = mse(D_REC_fake, torch.zeros_like(D_REC_fake))  # fake = 0
            D_REC_loss = D_REC_real_loss + D_REC_fake_loss

            fake_lowenergy = gen_LE(recombined)
            D_LE_real = disc_LE(lowenergy)
            D_LE_fake = disc_LE(fake_lowenergy.detach())
            D_LE_real_loss = mse(D_LE_real, torch.ones_like(D_LE_real))
            D_LE_fake_loss = mse(D_LE_fake, torch.zeros_like(D_LE_fake))
            D_LE_loss = D_LE_real_loss + D_LE_fake_loss

            # put it together
            D_loss = (D_REC_loss + D_LE_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_REC_fake = disc_REC(fake_recombined)
            D_LE_fake = disc_LE(fake_lowenergy)
            loss_G_REC = mse(D_REC_fake, torch.ones_like(D_REC_fake))
            loss_G_LE = mse(D_LE_fake, torch.ones_like(D_LE_fake))

            # cycle loss
            cycle_lowenergy = gen_LE(fake_recombined)
            cycle_recombined = gen_REC(fake_lowenergy)
            cycle_lowenergy_loss = l1(lowenergy, cycle_lowenergy)
            cycle_recombined_loss = l1(recombined, cycle_recombined)

            # identity loss
            identity_lowenergy = gen_LE(lowenergy)
            identity_recombined = gen_REC(recombined)
            identity_lowenergy_loss = l1(lowenergy, identity_lowenergy)
            identity_recombined_loss = l1(recombined, identity_recombined)

            # add all together
            G_loss = (loss_G_LE + loss_G_REC
                      + cycle_lowenergy_loss * cfg_trainer["LAMBDA_CYCLE"] + cycle_recombined_loss * cfg_trainer["LAMBDA_CYCLE"]
                      + identity_recombined_loss * cfg_trainer["LAMBDA_IDENTITY"] + identity_lowenergy_loss * cfg_trainer["LAMBDA_IDENTITY"])

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(REC_real=REC_reals / (idx + 1), REC_fake=REC_fakes / (idx + 1))

    return disc_REC, disc_LE, gen_LE, gen_REC


def eval_fn(gen_REC, loader, criterion, device, outputs_dir):
    gen_REC.eval()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    for idx, (lowenergy, recombined, file_names) in enumerate(loop):
        lowenergy = lowenergy.to(device)
        recombined = recombined.to(device)
        with torch.cuda.amp.autocast():
            fake_recombined = gen_REC(lowenergy.float())

            if outputs_dir:
                for output, file_name in zip(fake_recombined, file_names):
                    # SAVE OUTPUTS PNG
                    filename_output = "%s_output.png" % (file_name)
                    save_image(output, os.path.join(outputs_dir, filename_output))

        loss = criterion(fake_recombined, recombined)
        # statistics
        running_loss += loss.item() * lowenergy.size(0)
    epoch_loss = running_loss / len(loader)
    return epoch_loss


def train_cycle_gan(gen_REC, gen_LE, disc_REC, disc_LE, data_loaders, early_stopping_criterion, opt_disc, opt_gen, L1,
                    mse, model_fold_dir, cfg_trainer, device):
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best_gen_REC_wts = copy.deepcopy(gen_REC.state_dict())
    best_gen_LE_wts = copy.deepcopy(gen_LE.state_dict())
    best_disc_REC_wts = copy.deepcopy(disc_REC.state_dict())
    best_disc_LE_wts = copy.deepcopy(disc_LE.state_dict())
    best_loss = np.Inf


    history = {'train_loss': [], 'val_loss': []}

    epochs_no_improve = 0
    early_stop = False
    numero_di_epoche = 0
    for epoch in range(cfg_trainer["max_epochs"]):
        numero_di_epoche+=1
        # Train one epoch
        disc_REC, disc_LE, gen_LE, gen_REC = train_fn(disc_REC=disc_REC, disc_LE=disc_LE, gen_LE=gen_LE,
                                                      gen_REC=gen_REC,
                                                      loader=data_loaders['train'], opt_disc=opt_disc, opt_gen=opt_gen,
                                                      l1=L1,
                                                      mse=mse, d_scaler=d_scaler, g_scaler=g_scaler, device=device,
                                                      cfg_trainer=cfg_trainer)

        # Val
        epoch_loss = eval_fn(gen_REC=gen_REC, loader=data_loaders['val'], criterion=early_stopping_criterion, device=device, outputs_dir=False)
        epoch_train_loss = eval_fn(gen_REC=gen_REC, loader=data_loaders['train'], criterion=early_stopping_criterion,
                             device=device, outputs_dir=False)

        history['val_loss'].append(epoch_loss)
        history['train_loss'].append(epoch_train_loss)
        if epoch > cfg_trainer['warm_up']:
            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_gen_REC_wts = copy.deepcopy(gen_REC.state_dict())
                best_gen_LE_wts = copy.deepcopy(gen_LE.state_dict())
                best_disc_REC_wts = copy.deepcopy(disc_REC.state_dict())
                best_disc_LE_wts = copy.deepcopy(disc_LE.state_dict())
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
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Loss: {:4f}'.format(best_loss))

    gen_REC.load_state_dict(best_gen_REC_wts)
    gen_LE.load_state_dict(best_gen_LE_wts)
    disc_REC.load_state_dict(best_disc_REC_wts)
    disc_LE.load_state_dict(best_disc_LE_wts)

    # Save model
    save_checkpoint(gen_REC, opt_gen, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN_REC"]))
    save_checkpoint(gen_LE, opt_gen, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN_LE"]))
    save_checkpoint(disc_REC, opt_disc, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_CRITIC_REC"]))
    save_checkpoint(disc_LE, opt_disc, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_CRITIC_LE"]))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return gen_REC, gen_LE, disc_REC, disc_LE, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))


    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')


    #plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(model_plot_dir, "Loss.png"))
    plt.close()

def test_4metrics(gen_REC, loader_train, loader_val, loader_test, mse_metric, psnr_metric, device, outputs_dir,
                  save_output):
    gen_REC.eval()
    loop_test = tqdm(loader_test, leave=True)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_loss4 = 0.0

    if loader_train:
        loop_train = tqdm(loader_train, leave=True)
        with torch.no_grad():
            for idx, (lowenergy, recombined, file_names) in enumerate(loop_train):
                lowenergy = lowenergy.to(device)
                with torch.cuda.amp.autocast():
                    fake_recombined = gen_REC(lowenergy.float())
                    if save_output:
                        for output, file_name in zip(fake_recombined, file_names):  # SAVE OUTPUTS PNG
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    if loader_val:
        loop_val = tqdm(loader_val, leave=True)
        with torch.no_grad():
            for idx, (lowenergy, recombined, file_names) in enumerate(loop_val):
                lowenergy = lowenergy.to(device)
                with torch.cuda.amp.autocast():
                    fake_recombined = gen_REC(lowenergy.float())
                    if save_output:
                        for output, file_name in zip(fake_recombined, file_names):  # SAVE OUTPUTS PNG
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    with torch.no_grad():
        for idx, (lowenergy, recombined, file_names) in enumerate(loop_test):
            lowenergy = lowenergy.to(device)
            recombined = recombined.to(device)
            with torch.cuda.amp.autocast():
                fake_recombined = gen_REC(lowenergy.float())
                if save_output:
                    for output, file_name in zip(fake_recombined, file_names):  # SAVE OUTPUTS PNG
                        filename_output = "%s_output.png" % (file_name)
                        save_image(output, os.path.join(outputs_dir, filename_output))

            loss1 = mse_metric(fake_recombined.float(), recombined.float())
            loss2 = psnr_metric(fake_recombined.float(), recombined.float())
            loss3 = vif_p(fake_recombined.float(), recombined.float())
            loss4 = structural_similarity_index_measure(fake_recombined.float(), recombined.float(), data_range=1.0)
            #if platform == 'win32': # se sono su windows
            # loss4 = structural_similarity_index_measure(fake_recombined, recombined)
            #else: # se sono su alvis
            # loss4 = structural_similarity_index_measure(fake_recombined.to(torch.float16), recombined.to(torch.float16))

            # statistics
            running_loss1 += loss1.item() * lowenergy.size(0)
            running_loss2 += loss2.item() * lowenergy.size(0)
            running_loss3 += loss3.item() * lowenergy.size(0)
            running_loss4 += loss4.item() * lowenergy.size(0)

    epoch_loss1 = running_loss1 / len(loader_test)
    epoch_loss2 = running_loss2 / len(loader_test)
    epoch_loss3 = running_loss3 / len(loader_test)
    epoch_loss4 = running_loss4 / len(loader_test)
    return epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4

def test_cyclegan(gen_REC, loader_test, mse_metric, psnr_metric, device,
                  outputs_dir=None, save_outputs=False, *,
                  amp=True, lpips_backbone="alex", is_splits=10,
                  return_per_image=False):

    gen_REC.eval()

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

    lpips_fn = lpips.LPIPS(net=lpips_backbone).to(device).eval()

    inc_cls = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    inc_feat.fc = nn.Identity()

    logits_all = []
    feats_real_all = []
    feats_fake_all = []

    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else _nullcontext

    pbar = tqdm(loader_test, leave=True)
    for batch_idx, batch in enumerate(pbar):
        lowenergy, recombined, file_names = batch
        lowenergy = lowenergy.to(device, dtype=torch.float32)
        recombined = recombined.to(device, dtype=torch.float32)

        bsz = lowenergy.size(0)
        n_images += bsz

        with scaler_ctx():
            fake_recombined = gen_REC(lowenergy)

        if save_outputs and outputs_dir is not None:
            os.makedirs(outputs_dir, exist_ok=True)
            for out_img, name in zip(fake_recombined, file_names):
                save_image(out_img.clamp(0, 1),
                           os.path.join(outputs_dir, f"{name}_output.png"))

        mse_val = _to_scalar(mse_metric(fake_recombined, recombined))
        psnr_val = _to_scalar(psnr_metric(fake_recombined, recombined))

        with torch.cuda.amp.autocast(enabled=False):
            vif_val = _to_scalar(piq_vif(fake_recombined.float(), recombined.float()))
            ssim_val = _to_scalar(tm_ssim(fake_recombined, recombined, data_range=1.0))

        a_lp = _prep_for_lpips(fake_recombined)
        b_lp = _prep_for_lpips(recombined)
        lpips_val = _to_scalar(lpips_fn(a_lp, b_lp).mean())

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

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fake_inc = _prep_for_inception(fake_recombined).float().to(device)
                real_inc = _prep_for_inception(recombined).float().to(device)

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

    # Inception Score
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


# ------------- helper ----------------

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


