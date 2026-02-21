import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import numpy as np
import os
import yaml
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import collections
from torchmetrics import PeakSignalNoiseRatio
from sys import platform

from src.model.Unit.generator_model import Encoder, Generator, ResidualBlock
import src.model.Unit.util_model as util_model
import src.utils.util_general as util_general
import src.utils.util_data as util_data


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './model/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context


# --------------------------
# CONFIG
# --------------------------
if platform == 'win32':
    cfg_path = "./configs/unit_test_ale.yaml"
else:
    cfg_path = ".../configs/unit_test.yaml"

with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


exp_name = cfg['exp_name']
exp_name_excel = cfg['exp_name_excel']
print(exp_name_excel)
pretrained_path = os.path.join(cfg['data']['model_dir'], cfg['model']['pretrained_model'])
cv = cfg['data']['cv']
fold_list = list(range(cv))

device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print("Device:", device)


# --------------------------
# DIRS
# --------------------------
fold_dir = cfg['data']['fold_dir']
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)

save_img_output = cfg['output']['save_img_output']
if save_img_output:
    outputs_dir = os.path.join(report_dir, "outputs")
    util_general.create_dir(outputs_dir)
else:
    outputs_dir = None


# --------------------------
# METRICHE PER-FOLD
# --------------------------
fold_metrics = {
    "mse":   [],
    "psnr":  [],
    "vif":   [],
    "ssim":  [],
    "lpips": [],
    "fid":   [],
    "is":    []  # IS_mean
}


for fold in fold_list:
    print(f"\n===== TEST FOLD {fold} =====")

    # output fold
    if outputs_dir is not None:
        outputs_fold_dir = os.path.join(outputs_dir, str(fold))
        util_general.create_dir(outputs_fold_dir)
    else:
        outputs_fold_dir = None

    # --------------------------
    # TEST CSV / DATALOADER
    # --------------------------
    test_csv = os.path.join(fold_dir, str(fold), "test.csv")
    test_df = pd.read_csv(test_csv, index_col="id_slice")

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
        worker_init_fn=util_data.seed_worker
    )


    shared_dim = cfg['model']['dim'] * (2 ** cfg['model']['n_downsample'])
    shared_E = ResidualBlock(features=shared_dim).to(device)
    shared_G = ResidualBlock(features=shared_dim).to(device)

    E1 = Encoder(in_channels=1, dim=cfg['model']['dim'], n_downsample=cfg['model']['n_downsample'],
                 shared_block=shared_E).to(device)
    E2 = Encoder(in_channels=1, dim=cfg['model']['dim'], n_downsample=cfg['model']['n_downsample'],
                 shared_block=shared_E).to(device)
    G1 = Generator(out_channels=1, dim=cfg['model']['dim'], n_upsample=cfg['model']['n_downsample'],
                   shared_block=shared_G, out_activation='sigmoid').to(device)
    G2 = Generator(out_channels=1, dim=cfg['model']['dim'], n_upsample=cfg['model']['n_downsample'],
                   shared_block=shared_G, out_activation='sigmoid').to(device)

    opt_G = optim.Adam(
        list(E1.parameters()) + list(E2.parameters()) + list(G1.parameters()) + list(G2.parameters()),
        lr=cfg['trainer']['optimizer']['lr'], betas=(0.5, 0.999)
    )

    pretrained_model_dir = os.path.join(pretrained_path, str(fold))
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_E1']), E1, opt_G, 0.0001, device)
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_E2']), E2, opt_G, 0.0001, device)
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_G1']), G1, opt_G, 0.0001, device)
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_G2']), G2, opt_G, 0.0001, device)


    mse_metric  = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    results = util_model.test_unit(
        E1,  # encoder MRI
        G2,  # generator CT
        test_loader,  # loader_test
        mse_metric,
        psnr_metric,
        device,
        outputs_dir=outputs_fold_dir,
        save_outputs=save_img_output,
        return_per_image=True
    )

    fold_metrics["mse"].append(results["mse"])
    fold_metrics["psnr"].append(results["psnr"])
    fold_metrics["vif"].append(results["vif"])
    fold_metrics["ssim"].append(results["ssim"])
    fold_metrics["lpips"].append(results["lpips"])
    fold_metrics["fid"].append(results["fid"])
    fold_metrics["is"].append(results["is_mean"])

    print(f"[fold {fold}] mse={results['mse']:.6f} psnr={results['psnr']:.4f} "
          f"vif={results['vif']:.4f} ssim={results['ssim']:.4f} "
          f"lpips={results['lpips']:.4f} fid={results['fid']:.4f} is_mean={results['is_mean']:.4f}")



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

MSE_0, MSE_1, MSE_2, MSE_3, MSE_4       = fold_metrics["mse"]
PSNR_0,PSNR_1,PSNR_2,PSNR_3,PSNR_4      = fold_metrics["psnr"]
VIF_0, VIF_1, VIF_2, VIF_3, VIF_4       = fold_metrics["vif"]
SSIM_0,SSIM_1,SSIM_2,SSIM_3,SSIM_4      = fold_metrics["ssim"]
LPIPS_0,LPIPS_1,LPIPS_2,LPIPS_3,LPIPS_4 = fold_metrics["lpips"]
FID_0,FID_1,FID_2,FID_3,FID_4           = fold_metrics["fid"]
IS_0,IS_1,IS_2,IS_3,IS_4                = fold_metrics["is"]

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
    IS_0, IS_1, IS_2, IS_3, IS_4
]

ordered_cols = [
    "MSE_m","MSE_sem",
    "PSNR_m","PSNR_sem",
    "VIF_m","VIF_sem",
    "SSIM_m","SSIM_sem",
    "LPIPS_m","LPIPS_sem",
    "FID_m","FID_sem",
    "IS_m","IS_sem",

    "MSE_0","MSE_1","MSE_2","MSE_3","MSE_4",
    "PSNR_0","PSNR_1","PSNR_2","PSNR_3","PSNR_4",
    "VIF_0","VIF_1","VIF_2","VIF_3","VIF_4",
    "SSIM_0","SSIM_1","SSIM_2","SSIM_3","SSIM_4",
    "LPIPS_0","LPIPS_1","LPIPS_2","LPIPS_3","LPIPS_4",
    "FID_0","FID_1","FID_2","FID_3","FID_4",
    "IS_0","IS_1","IS_2","IS_3","IS_4"
]

df_out = pd.DataFrame([ordered_row], columns=ordered_cols)

excel_path = os.path.join(report_dir, f"{exp_name_excel}_test_summary.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="metrics", index=False)

print(f"\n[OK] Excel saved in: {excel_path}")
