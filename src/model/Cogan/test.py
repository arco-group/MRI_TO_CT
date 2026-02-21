import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os, yaml, ssl
import torch
import torch.nn as nn
import pandas as pd
from sys import platform
from torchmetrics import PeakSignalNoiseRatio

from src.model.Cogan.generator_model import CoupledGenerator
import model.Cogan.codici_vecchi.util_model_vecchio as util_model
import src.utils.util_data as util_data
import src.utils.util_general as util_general

ssl._create_default_https_context = ssl._create_unverified_context
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './model/cnn/pretrained'

# --------------------------
# Config
# --------------------------
if platform == 'win32':
    cfg_path = "./configs/cogan_test.yaml"
else:
    cfg_path = ".../configs/cogan_test.yaml"

with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print("Device:", device)

exp_name = cfg["exp_name"]
exp_name_excel = cfg['exp_name_excel']
print(exp_name_excel)
model_name = cfg["model"]["model_name"]
cv = cfg["data"]["cv"]
fold_list = list(range(cv))

fold_dir = cfg["data"]["fold_dir"]
pretrained_path = os.path.join(cfg["data"]["model_dir"], cfg["model"]["pretrained_model"])

report_dir = os.path.join(cfg["data"]["report_dir"], exp_name)
util_general.create_dir(report_dir)

save_img_output = cfg.get("output", {}).get("save_img_output", True)
outputs_dir = os.path.join(report_dir, "outputs") if save_img_output else None
if outputs_dir:
    util_general.create_dir(outputs_dir)

# --------------------------
# Loop fold
# --------------------------
fold_metrics = []

for fold in fold_list:
    print(f"\n===== TEST FOLD {fold} =====")

    outputs_fold_dir = os.path.join(outputs_dir, str(fold)) if outputs_dir else None
    if outputs_fold_dir:
        util_general.create_dir(outputs_fold_dir)

    # ---- test.csv ----
    test_csv = os.path.join(fold_dir, str(fold), "test.csv")
    fold_data_test = pd.read_csv(test_csv, index_col="id_slice")

    dataset_test = util_data.ImgDataset(
        data=fold_data_test,
        cfg_data=cfg["data"],
        step="test",
        do_augmentation=False
    )

    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=num_workers, worker_init_fn=util_data.seed_worker
    )

    # ---- Model CoGAN ----
    G = CoupledGenerator(in_ch=1).to(device)


    ckpt_path = os.path.join(pretrained_path, str(fold), cfg["trainer"]["CHECKPOINT_G"])
    util_general.load_checkpoint(
        ckpt_path,
        G,
        optimizer=None,
        lr=None,
        map_location=device
    )

    G.eval()

    # ---- Metrics ----
    mse_metric  = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    results = util_model.test_cogan(
        G, loader_test,
        mse_metric=mse_metric,
        psnr_metric=psnr_metric,
        device=device,
        outputs_dir=outputs_fold_dir,
        save_outputs=bool(outputs_fold_dir),
        amp=True,
        lpips_backbone="alex",
        is_splits=10,
        return_per_image=False
    )

    print(
        f"Fold {fold} → "
        f"MSE={results['mse']:.6f}, PSNR={results['psnr']:.4f}, "
        f"VIF={results['vif']:.4f}, SSIM={results['ssim']:.4f}, "
        f"LPIPS={results['lpips']:.4f}, FID={results['fid']:.4f}, "
        f"IS={results['is_mean']:.4f}±{results['is_std']:.4f}"
    )

    fold_metrics.append({
        "fold": fold,
        "mse": results["mse"],
        "psnr": results["psnr"],
        "vif": results["vif"],
        "ssim": results["ssim"],
        "lpips": results["lpips"],
        "fid": results["fid"],
        "is_mean": results["is_mean"],
        "is_std": results["is_std"],
    })



df_folds = pd.DataFrame(fold_metrics).set_index("fold").sort_index()

mse_m, mse_std = df_folds["mse"].mean(), df_folds["mse"].std(ddof=1)
psnr_m, psnr_std = df_folds["psnr"].mean(), df_folds["psnr"].std(ddof=1)
vif_m, vif_std = df_folds["vif"].mean(), df_folds["vif"].std(ddof=1)
ssim_m, ssim_std = df_folds["ssim"].mean(), df_folds["ssim"].std(ddof=1)
lpips_m, lpips_std = df_folds["lpips"].mean(), df_folds["lpips"].std(ddof=1)
fid_m, fid_std = df_folds["fid"].mean(), df_folds["fid"].std(ddof=1)
is_m, is_std = df_folds["is_mean"].mean(), df_folds["is_mean"].std(ddof=1)

def fcol(metric, i): return float(df_folds.loc[i, metric])

row_out = {
    "MSE_m": mse_m, "MSE_std": mse_std,
    "PSNR_m": psnr_m, "PSNR_std": psnr_std,
    "VIF_m": vif_m, "VIF_std": vif_std,
    "SSIM_m": ssim_m, "SSIM_std": ssim_std,
    "LPIPS_m": lpips_m, "LPIPS_std": lpips_std,
    "FID_m": fid_m, "FID_std": fid_std,
    "IS_m": is_m, "IS_std": is_std,
}

for i in range(cv): row_out[f"MSE_{i}"] = fcol("mse", i)
for i in range(cv): row_out[f"PSNR_{i}"] = fcol("psnr", i)
for i in range(cv): row_out[f"VIF_{i}"] = fcol("vif", i)
for i in range(cv): row_out[f"SSIM_{i}"] = fcol("ssim", i)
for i in range(cv): row_out[f"LPIPS_{i}"] = fcol("lpips", i)
for i in range(cv): row_out[f"FID_{i}"] = fcol("fid", i)
for i in range(cv): row_out[f"IS_{i}"] = fcol("is_mean", i)

ordered_cols = [
    "MSE_m","MSE_std","PSNR_m","PSNR_std","VIF_m","VIF_std","SSIM_m","SSIM_std",
    "LPIPS_m","LPIPS_std","FID_m","FID_std","IS_m","IS_std",
    "MSE_0","MSE_1","MSE_2","MSE_3","MSE_4",
    "PSNR_0","PSNR_1","PSNR_2","PSNR_3","PSNR_4",
    "VIF_0","VIF_1","VIF_2","VIF_3","VIF_4",
    "SSIM_0","SSIM_1","SSIM_2","SSIM_3","SSIM_4",
    "LPIPS_0","LPIPS_1","LPIPS_2","LPIPS_3","LPIPS_4",
    "FID_0","FID_1","FID_2","FID_3","FID_4",
    "IS_0","IS_1","IS_2","IS_3","IS_4",
]

df_out = pd.DataFrame([[row_out[c] for c in ordered_cols]], columns=ordered_cols)

excel_path = os.path.join(report_dir, f"{exp_name_excel}_test.xlsx")
df_out.to_excel(excel_path, index=False)
print("\n[OK] Excel (1 riga) salvato in:", excel_path)
