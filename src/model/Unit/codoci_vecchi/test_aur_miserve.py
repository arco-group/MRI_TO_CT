import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])  # permette import relativi al progetto
import os
import yaml
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchmetrics import PeakSignalNoiseRatio
from src.models.unit.generator_model import Encoder, Generator, ResidualBlock
import src.models.unit.util_model as util_model
import src.utils.util_general as util_general
import src.utils.util_data as util_data


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './model/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context


args = {}
args['cfg_file'] = "./configs/unit_test.yaml"
with open(args['cfg_file']) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Parametri base
exp_name = cfg['exp_name']
model_name = cfg['model']['model_name']
pretrained_path = os.path.join(cfg['data']['model_dir'], cfg['model']['pretrained_model'])
#cv = cfg['data']['cv']
#fold_list = list(range(cv))
fold_list = [0]

device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f"Device: {device}")


# Cartelle
fold_dir = cfg['data']['fold_dir']
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)
save_img_output = cfg['output']['save_img_output']
if save_img_output:
    outputs_dir = os.path.join(report_dir, "outputs")
    util_general.create_dir(outputs_dir)
else:
    outputs_dir = False

# Inizializzo file per le metriche
all_fold_rows = []

for fold in fold_list:
    print(f"\n===== TEST FOLD {fold} =====")

    outputs_fold_dir = os.path.join(outputs_dir, str(fold)) if save_img_output else False
    if outputs_fold_dir:
        util_general.create_dir(outputs_fold_dir)

    # Dataloaders
    fold_data = {step: pd.read_csv(os.path.join(fold_dir, str(fold), '%s.csv' % step), index_col='id') for step in
                 ['train', 'val', 'test']}
    datasets = {step: util_data.ImgDataset(data=fold_data[step], cfg_data=cfg['data'], step=step, do_augmentation=True)
                for step in ['train', 'val', 'test']}
    data_loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True,
                                             num_workers=num_workers, worker_init_fn=util_data.seed_worker),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False,
                                           num_workers=num_workers, worker_init_fn=util_data.seed_worker),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=num_workers,
                                            worker_init_fn=util_data.seed_worker)}

    # -------------------------------------------------------------
    # Caricamento modelli e pesi pre-addestrati
    # -------------------------------------------------------------
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

    opt_G = optim.Adam(list(E1.parameters()) + list(E2.parameters()) + list(G1.parameters()) + list(G2.parameters()),
                       lr=cfg['trainer']['optimizer']['lr'], betas=(0.5, 0.999))

    pretrained_model_dir = os.path.join(pretrained_path, str(fold))
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_E1']), E1, opt_G, 0.0001, device)
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_E2']), E2, opt_G, 0.0001, device)
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_G1']), G1, opt_G, 0.0001, device)
    util_general.load_checkpoint(os.path.join(pretrained_model_dir, cfg['trainer']['CHECKPOINT_G2']), G2, opt_G, 0.0001, device)

    # -------------------------------------------------------------
    # Test metriche
    # -------------------------------------------------------------
    mse = nn.MSELoss().to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    results = util_model.test_unit(E1, G2, loader_test=data_loaders['test'], mse_metric=mse, psnr_metric=psnr,
                                   device=device, outputs_dir=outputs_fold_dir, save_outputs=save_img_output)

    print("mse_test_results:", results['mse'], "psnr_test_results:", results['psnr'], "vif_test_results:",
          results['vif'], "ssim_test_results:", results['ssim'], "lpips:", results['lpips'], "ms_ssim:",
          results['ms_ssim'], "fid:", results['fid'])

    # --- NEW: normalizza e salva la riga di metriche per questo fold
    row = {"fold": fold}
    for k, v in results.items():
        # converti eventuali tensori a float
        if torch.is_tensor(v):
            v = v.detach().item()
        # prova conversione numerica, altrimenti lascia com'è
        try:
            row[k] = float(v)
        except (TypeError, ValueError):
            row[k] = v
    all_fold_rows.append(row)

# --- NEW: dopo il loop su tutte le fold, costruisci il DataFrame e salva in Excel
if len(all_fold_rows) > 0:
    df = pd.DataFrame(all_fold_rows).set_index("fold").sort_index()

    # calcola mean e std solo sulle colonne numeriche
    mean_row = df.select_dtypes(include="number").mean()
    std_row = df.select_dtypes(include="number").std(ddof=1)

    # aggiungi come righe finali
    df_out = pd.concat([df, pd.DataFrame([mean_row, std_row], index=["mean", "std"])], axis=0)

    # salva Excel
    excel_path = os.path.join(report_dir, f"{exp_name}_test.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="metrics", index=True)

    print(f"[OK] Metriche salvate in: {excel_path}")
else:
    print("[WARN] Nessuna metrica raccolta: controlla che 'results' venga popolato in ogni fold.")

