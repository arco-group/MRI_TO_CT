import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import yaml
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchmetrics import PeakSignalNoiseRatio
from sys import platform as sys_platform

from src.model.Bicyclegan.generator_model import Generator, Encoder
import src.model.Bicyclegan.util_model as util_model
import src.utils.util_general as util_general
import src.utils.util_data as util_data
from src.utils.util_general import load_checkpoint


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './model/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context


# --------------------------
# CONFIG
# --------------------------
if sys_platform == 'win32':
    cfg_file = "./configs/bicyclegan_test.yaml"
else:
    cfg_file = "/mimer/NOBACKUP/groups/snic2022-5-277/rofena/Progetto_Alessandro/configs/bicyclegan_test.yaml"

with open(cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


exp_name        = cfg['exp_name']
model_name      = cfg['model']['model_name']
exp_name_excel = cfg['exp_name_excel'] #todo
print(exp_name_excel)
pretrained_path = os.path.join(cfg['data']['model_dir'], cfg['model']['pretrained_model'])
learning_rate   = cfg['trainer']['optimizer']['lr']
cv              = cfg['data']['cv']
fold_list       = list(range(cv))

save_img_output = cfg['output']['save_img_output']


# --------------------------
# DEVICE
# --------------------------
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print("Device:", device)


# --------------------------
# DIRS
# --------------------------
fold_dir   = cfg['data']['fold_dir']
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)

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
    "is":    []
}


for fold in fold_list:
    print(f"\n===== TEST Fold {fold} =====")

    # output fold
    if outputs_dir is not None:
        outputs_fold_dir = os.path.join(outputs_dir, str(fold))
        util_general.create_dir(outputs_fold_dir)
    else:
        outputs_fold_dir = None

    # --------------------------
    # SOLO TEST DATALOADER
    # --------------------------
    test_csv = os.path.join(fold_dir, str(fold), "test.csv")
    test_df  = pd.read_csv(test_csv, index_col="id_slice")

    test_dataset = util_data.ImgDataset(
        data=test_df,
        cfg_data=cfg['data'],
        step="test",
        do_augmentation=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=num_workers, worker_init_fn=util_data.seed_worker
    )

    # --------------------------
    # MODELLI + PESI
    # --------------------------
    img_dim    = cfg['data']['img_dim']
    latent_dim = cfg['model']['latent_dim'] if 'latent_dim' in cfg['model'] else 8  # fallback

    generator = Generator(
        latent_dim=latent_dim,
        img_height=img_dim,
        img_width=img_dim,
        in_channels=1,
        out_channels=1,
        out_activation='sigmoid',
    ).to(device)

    encoder = Encoder(latent_dim=latent_dim, in_channels=1).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_E = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    pretrained_model_dir = os.path.join(pretrained_path, str(fold))
    CHECKPOINT_GEN = os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_GEN'])
    CHECKPOINT_ENC = os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_ENC'])

    load_checkpoint(CHECKPOINT_GEN, generator, opt_G, learning_rate, device)
    load_checkpoint(CHECKPOINT_ENC, encoder, opt_E, learning_rate, device)

    # --------------------------
    # TEST METRICHE
    # --------------------------
    mse_metric  = nn.MSELoss().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    results = util_model.test_bicyclegan(
        generator=generator,
        encoder=encoder,
        loader_test=test_loader,
        mse_metric=mse_metric,
        psnr_metric=psnr_metric,
        device=device,
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


# --------------------------
# EXCEL 1 RIGA ORDINATA
# --------------------------
def _mean_std(x):
    s = pd.Series(x, dtype="float64")
    return float(s.mean()), float(s.std(ddof=1))

MSE_m,   MSE_std   = _mean_std(fold_metrics["mse"])
PSNR_m,  PSNR_std  = _mean_std(fold_metrics["psnr"])
VIF_m,   VIF_std   = _mean_std(fold_metrics["vif"])
SSIM_m,  SSIM_std  = _mean_std(fold_metrics["ssim"])
LPIPS_m, LPIPS_std = _mean_std(fold_metrics["lpips"])
FID_m,   FID_std   = _mean_std(fold_metrics["fid"])
IS_m,    IS_std    = _mean_std(fold_metrics["is"])

MSE_0, MSE_1, MSE_2, MSE_3, MSE_4       = fold_metrics["mse"]
PSNR_0,PSNR_1,PSNR_2,PSNR_3,PSNR_4      = fold_metrics["psnr"]
VIF_0, VIF_1, VIF_2, VIF_3, VIF_4       = fold_metrics["vif"]
SSIM_0,SSIM_1,SSIM_2,SSIM_3,SSIM_4      = fold_metrics["ssim"]
LPIPS_0,LPIPS_1,LPIPS_2,LPIPS_3,LPIPS_4 = fold_metrics["lpips"]
FID_0,FID_1,FID_2,FID_3,FID_4           = fold_metrics["fid"]
IS_0,IS_1,IS_2,IS_3,IS_4                = fold_metrics["is"]

ordered_row = [
    MSE_m, MSE_std,
    PSNR_m, PSNR_std,
    VIF_m, VIF_std,
    SSIM_m, SSIM_std,
    LPIPS_m, LPIPS_std,
    FID_m, FID_std,
    IS_m, IS_std,

    MSE_0, MSE_1, MSE_2, MSE_3, MSE_4,
    PSNR_0, PSNR_1, PSNR_2, PSNR_3, PSNR_4,
    VIF_0, VIF_1, VIF_2, VIF_3, VIF_4,
    SSIM_0, SSIM_1, SSIM_2, SSIM_3, SSIM_4,
    LPIPS_0, LPIPS_1, LPIPS_2, LPIPS_3, LPIPS_4,
    FID_0, FID_1, FID_2, FID_3, FID_4,
    IS_0, IS_1, IS_2, IS_3, IS_4
]

ordered_cols = [
    "MSE_m","MSE_std",
    "PSNR_m","PSNR_std",
    "VIF_m","VIF_std",
    "SSIM_m","SSIM_std",
    "LPIPS_m","LPIPS_std",
    "FID_m","FID_std",
    "IS_m","IS_std",

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

print(f"\n[OK] Excel salvato (1 riga ordinata) in: {excel_path}")
