import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import ssl
import yaml
import torch
import torch.nn as nn
import pandas as pd
from sys import platform
from torchmetrics import PeakSignalNoiseRatio
import numpy as np

from src.model.Pix2Pix.generator_model import Generator
import src.model.Pix2Pix.util_model as util_model
import src.utils.util_general as util_general
import src.utils.util_data as util_data


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './model/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context

# CONFIG
if platform == 'win32':  #windows
    args = {}
    args['cfg_file'] = "./configs/pix2pix_test.yaml"
    with open(args['cfg_file']) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
else:  # others
    args = {}
    args['cfg_file'] = ".../configs/pix2pix_test.yaml"
    with open(args['cfg_file']) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

# Parameters
exp_name = cfg['exp_name']
exp_name_excel = cfg['exp_name_excel']
print(exp_name_excel)
model_name = cfg['model']['model_name']
pretrained_path = os.path.join(cfg['data']['model_dir'], cfg['model']['pretrained_model'])
learning_rate = cfg['trainer']['optimizer']['lr']
cv = cfg['data']['cv']
fold_list = list(range(cv))
save_img_output = cfg['output']['save_img_output']

# DEVICE
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
fold_dir = cfg['data']['fold_dir']
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)

if save_img_output:
    outputs_dir = os.path.join(report_dir, "outputs")
    util_general.create_dir(outputs_dir)
else:
    outputs_dir = None


# METRICS FOLD
fold_metrics = {
    "mse":   [],
    "psnr":  [],
    "vif":   [],
    "ssim":  [],
    "lpips": [],
    "fid":   [],
    "is":    [],   # Inception Score (mean)
}


per_image_metrics_all_folds = []


for fold in fold_list:
    print(f"\n===== TEST FOLD {fold} =====")


    if save_img_output:
        outputs_fold_dir = os.path.join(outputs_dir, str(fold))
        util_general.create_dir(outputs_fold_dir)
    else:
        outputs_fold_dir = False


    test_csv = os.path.join(fold_dir, str(fold), "test.csv")
    test_df = pd.read_csv(test_csv, index_col='id_slice')

    test_dataset = util_data.ImgDataset(
        data=test_df,
        cfg_data=cfg['data'],
        step="test",
        do_augmentation=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=util_data.seed_worker,
    )


    gen = Generator(in_channels=1, features=64).to(device)


    opt_gen = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    pretrained_model_dir = os.path.join(pretrained_path, str(fold))
    CHECKPOINT_GEN = os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_GEN'])
    util_general.load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, learning_rate, device)

    # METRICS
    mse_metric  = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    results = util_model.test_pix2pix(
        gen,
        loader_test=test_loader,
        mse_metric=mse_metric,
        psnr_metric=psnr_metric,
        device=device,
        outputs_dir=outputs_fold_dir,
        save_outputs=save_img_output,
        return_per_image=True,
    )


    fold_metrics["mse"].append(results["mse"])
    fold_metrics["psnr"].append(results["psnr"])
    fold_metrics["vif"].append(results["vif"])
    fold_metrics["ssim"].append(results["ssim"])
    fold_metrics["lpips"].append(results["lpips"])
    fold_metrics["fid"].append(results["fid"])
    fold_metrics["is"].append(results["is_mean"])


    per_image_metrics_all_folds.append(results["per_image"])

    print(
        f"[fold {fold}]",
        "mse:", results["mse"],
        "psnr:", results["psnr"],
        "vif:", results["vif"],
        "ssim:", results["ssim"],
        "lpips:", results["lpips"],
        "fid:", results["fid"],
        "is_mean:", results["is_mean"],
        "is_std:", results["is_std"],
    )


def _mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    sem  = float(x.std(ddof=1) / np.sqrt(len(x)))
    return mean, sem



MSE_m,   MSE_sem   = _mean_sem(fold_metrics["mse"])
PSNR_m,  PSNR_sem  = _mean_sem(fold_metrics["psnr"])
VIF_m,   VIF_sem   = _mean_sem(fold_metrics["vif"])
SSIM_m,  SSIM_sem  = _mean_sem(fold_metrics["ssim"])
LPIPS_m, LPIPS_sem = _mean_sem(fold_metrics["lpips"])
FID_m,   FID_sem   = _mean_sem(fold_metrics["fid"])
IS_m,    IS_sem    = _mean_sem(fold_metrics["is"])



MSE_0, MSE_1, MSE_2, MSE_3, MSE_4        = fold_metrics["mse"]
PSNR_0, PSNR_1, PSNR_2, PSNR_3, PSNR_4   = fold_metrics["psnr"]
VIF_0, VIF_1, VIF_2, VIF_3, VIF_4        = fold_metrics["vif"]
SSIM_0, SSIM_1, SSIM_2, SSIM_3, SSIM_4   = fold_metrics["ssim"]
LPIPS_0, LPIPS_1, LPIPS_2, LPIPS_3, LPIPS_4 = fold_metrics["lpips"]
FID_0, FID_1, FID_2, FID_3, FID_4        = fold_metrics["fid"]
IS_0, IS_1, IS_2, IS_3, IS_4             = fold_metrics["is"]


ordered_row = [
    MSE_m, MSE_sem,
    PSNR_m, PSNR_sem,
    VIF_m, VIF_sem,
    SSIM_m, SSIM_sem,
    LPIPS_m, LPIPS_sem,
    FID_m, FID_sem,
    IS_m, IS_sem,

    MSE_0, MSE_1, MSE_2, MSE_3, MSE_4,
    PSNR_0, PSNR_1, PSNR_2, PSNR_3, PSNR_4,
    VIF_0, VIF_1, VIF_2, VIF_3, VIF_4,
    SSIM_0, SSIM_1, SSIM_2, SSIM_3, SSIM_4,
    LPIPS_0, LPIPS_1, LPIPS_2, LPIPS_3, LPIPS_4,
    FID_0, FID_1, FID_2, FID_3, FID_4,
    IS_0, IS_1, IS_2, IS_3, IS_4,
]

ordered_cols = [
    "MSE_m", "MSE_sem",
    "PSNR_m", "PSNR_sem",
    "VIF_m", "VIF_sem",
    "SSIM_m", "SSIM_sem",
    "LPIPS_m", "LPIPS_sem",
    "FID_m", "FID_sem",
    "IS_m", "IS_sem",

    "MSE_0", "MSE_1", "MSE_2", "MSE_3", "MSE_4",
    "PSNR_0", "PSNR_1", "PSNR_2", "PSNR_3", "PSNR_4",
    "VIF_0", "VIF_1", "VIF_2", "VIF_3", "VIF_4",
    "SSIM_0", "SSIM_1", "SSIM_2", "SSIM_3", "SSIM_4",
    "LPIPS_0", "LPIPS_1", "LPIPS_2", "LPIPS_3", "LPIPS_4",
    "FID_0", "FID_1", "FID_2", "FID_3", "FID_4",
    "IS_0", "IS_1", "IS_2", "IS_3", "IS_4",
]

df_out = pd.DataFrame([ordered_row], columns=ordered_cols)

excel_path = os.path.join(report_dir, f"{exp_name_excel}_test_summary.xlsx") #todo
df_out.to_excel(excel_path, sheet_name="metrics", index=False)

print(f"[OK] Excel salvato (1 riga ordinata) in: {excel_path}")
