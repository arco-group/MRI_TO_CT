import torch
# import torchvision.transforms as T
import numpy as np
import cv2
import os
import json
import imutils
import random
# from torchvision.io import read_image
from scipy.ndimage import shift
import nibabel as nib


# -------------------------
# Utility & randomness
# -------------------------
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -------------------------
# I/O
# -------------------------
def read_nii(nii_path):
    nii_volume = nib.load(nii_path)
    volume = nii_volume.get_fdata()
    if volume.ndim > 3:
        volume = volume[..., 0]
    return volume  # (H, W, slices) or (H, W)

def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out without changing final dimensions.

    # 
    #Center zoom in/out of the given image and returning an enlarged/shrinked view of
    #the image without changing dimensions
    #Args:
    #    img : Image array
    #    zoom_factor : amount of zoom as a ratio (0 to Inf)
    #
    """
    height, width = img.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    # bbox in resized coords
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    # map back to original
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1 = (height - resize_height) // 2
    pad_width1 = (width - resize_width) // 2
    pad_height2 = (height - resize_height) - pad_height1
    pad_width2 = (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augmentation(img_x, img_y):
    # shift
    r = random.randint(0, 100)
    if r > 70:
        shift_perc = 0.1
        r1 = random.randint(-int(shift_perc * img_x.shape[0]), int(shift_perc * img_x.shape[0]))
        r2 = random.randint(-int(shift_perc * img_x.shape[1]), int(shift_perc * img_x.shape[1]))
        img_x = shift(img_x, [r1, r2], mode='nearest')
        img_y = shift(img_y, [r1, r2], mode='nearest')
    # zoom
    r = random.randint(0, 100)
    if r > 70:
        zoom_perc = 0.1
        zoom_factor = random.uniform(1 - zoom_perc, 1 + zoom_perc)
        img_x = clipped_zoom(img_x, zoom_factor=zoom_factor)
        img_y = clipped_zoom(img_y, zoom_factor=zoom_factor)
    # flip
    r = random.randint(0, 100)
    if r > 70:
        img_x = cv2.flip(img_x, 1)
        img_y = cv2.flip(img_y, 1)
    # rotation
    r = random.randint(0, 100)
    if r > 70:
        max_angle = 15
        r = random.randint(-max_angle, max_angle)
        img_x = imutils.rotate(img_x, r)
        img_y = imutils.rotate(img_y, r)
    return img_x, img_y


# -------------------------
# Intensity helpers (robusti)
# -------------------------



def normalize_01(img: np.ndarray, img_path) -> np.ndarray:

    finite = np.isfinite(img)
    if not np.any(finite):
        print("L'immagine NaN/inf ha questo percorso:" + img_path)
        return np.zeros_like(img, dtype=np.float32)

    vmin = np.min(img[finite])
    vmax = np.max(img[finite])

    if vmax <= vmin:
        print("L'immagine ha tutti i valori costanti:" + img_path)
        out = np.zeros_like(img, dtype=np.float32)
        out[~finite] = 0.0
        return out

    out = (img - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.astype(np.float32)


# -------------------------
# Padding
# -------------------------
def padding(img):

    h, w = img.shape[:2]
    if h == w:
        return img
    min_value = np.min(img)
    if h > w:
        total = h - w
        left = total // 2
        right = total - left
        pad_spec = ((0, 0), (left, right))
    else:
        total = w - h
        top = total // 2
        bottom = total - top
        pad_spec = ((top, bottom), (0, 0))

    return np.pad(img, pad_spec, mode='constant', constant_values=min_value)

def contrast_stretching(img, ww, wc):

    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    return np.clip(img, lo, hi)

# -------------------------
# Loader
# -------------------------
def loader(img_x_path, img_y_path, img_dim, do_augmentation, wc, ww, step="train"):
    # 1) lettura
    img_x = read_nii(img_x_path)#legge il percorso e restituisce l'immagine, carica l'immagine per poter fare il pre-processing successivo
    img_y = read_nii(img_y_path)

    # 2) padding a quadrato
    #img_x = padding(img_x)
    #img_y = padding(img_y)

    # 3) contrast stretching / windowing su Y (CT)
    #    (puoi cambiare wc/ww in base al distretto clinico)
    #img_y = contrast_stretching(img_y, wc=wc, ww=ww)
    # AB --> wc = 50, ww = 400
    # TH --> wc = 40, ww = 400
    # HN --> wc = 40, ww = 80

    # 4) normalizzazione robusta in [0,1]
    #img_x = normalize_01(img_x, img_x_path)
    #img_y = normalize_01(img_y, img_y_path)

    # 5) resize consistente
    #img_x = cv2.resize(img_x.astype(np.float32), dsize=(img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
    #img_y = cv2.resize(img_y.astype(np.float32), dsize=(img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
    #Elaborazione immagine con OpenCV per ridimensionare l'immagine. "img_x.astype(np.float32)" converte la matrice
    # numpy in una matrice di numeri float, "dsize=(img_dim, img_dim)" dimensione in larghezza e altezza dell'immagine,
    #Imposta le dimensioni finali dell'immagine (nel nostro caso 256) si vede dal file yamal,
    # "interpolation=cv2.INTER_LINEAR" è il metodo di interpolazione per calcolare i nuovi valori dei pixel

    # 6) augmentation (solo in train)
    if do_augmentation and str(step).lower() == "train": #modifica
        img_x, img_y = augmentation(img_x, img_y) #modifica

    #img_x = (img_x * 2.0) - 1.0 #queste le devo decomentare solo per dualgan
    #img_y = (img_y * 2.0) - 1.0 #queste le devo decommentare solo per dualgan

    # 7) to torch tensor (C,H,W) con canale singolo
    #img_x = torch.from_numpy(img_x).unsqueeze(0)  # (1, H, W)
    #img_y = torch.from_numpy(img_y).unsqueeze(0)  # (1, H, W)
    img_x = torch.from_numpy(img_x).unsqueeze(0).float()  # (1, H, W) - forza float32
    img_y = torch.from_numpy(img_y).unsqueeze(0).float()  # (1, H, W) - forza float32


    return img_x, img_y


# -------------------------
# Dataset
# -------------------------
class ImgDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, cfg_data, step, do_augmentation):
        'Initialization'
        self.step = step
        self.img_dir = cfg_data["img_dir"]
        self.data = data
        # Dim
        self.img_dim = cfg_data["img_dim"]
        # Augmentation
        self.do_augmentation = do_augmentation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        img_x_path = row.slice_mri_path
        img_y_path = row.slice_ct_path
        # Load data
        img_x_path = img_x_path.replace("\\", "/")
        img_y_path = img_y_path.replace("\\", "/")
        wc = row.wc
        ww = row.ww

        img_x, img_y = loader(img_x_path=img_x_path, img_y_path=img_y_path, img_dim=self.img_dim, do_augmentation=self.do_augmentation, wc = wc, ww = ww, step=self.step)

        return img_x, img_y, id


