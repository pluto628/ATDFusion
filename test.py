import torch
import os
import numpy as np
import cv2
from PIL import Image
import sys
import warnings
import logging
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.reins_dinov2 import get_std_reins_dinov2_large
from encoder2 import FeatureExtractor
from decoder import Decoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
dataset_name = "MRI-CT"
model_path = "checkpoints/MRI-CT.pth"
ir_folder = "Medical_dataset_test/MRI-CT/CT"
vis_folder = "Medical_dataset_test/MRI-CT/MRI"
# ===============================

model_dir = os.path.dirname(model_path)
save_base = os.path.join(os.path.dirname(model_dir), "test_result")
model_name = os.path.splitext(os.path.basename(model_path))[0]
test_out_folder = os.path.join(save_base, f"{dataset_name}")
os.makedirs(test_out_folder, exist_ok=True)

checkpoint = torch.load(model_path, map_location=device)
model1 = get_std_reins_dinov2_large().to(device)
model2 = Decoder().to(device)
Encoder2 = FeatureExtractor(device=device)
model1.load_state_dict(checkpoint['Encoder1_state_dict'])
model2.load_state_dict(checkpoint['decoder_state_dict'])

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    if mode == 'RGB':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        return np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

ir_images = sorted(os.listdir(ir_folder))
vis_images = sorted(os.listdir(vis_folder))
if len(ir_images) != len(vis_images):
    print("Image count mismatch")
    exit()
else:
    print(f"Image count matched, total {len(ir_images)} image pairs")

def prep_data(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
    return torch.tensor(np.transpose(np.expand_dims(img / 255.0, 0), (0, 3, 1, 2)), dtype=torch.float32).to(device)

with torch.no_grad():
    for img_name in tqdm(ir_images, desc="Processing", unit="images"):
        if dataset_name == "MRI-CT":
            ir_img = image_read_cv2(os.path.join(ir_folder, img_name), mode='GRAY')
            vis_img = image_read_cv2(os.path.join(vis_folder, img_name), mode='GRAY')

            data_IR = prep_data(ir_img)
            data_VIS = prep_data(vis_img)

            feat1 = model1(data_IR)
            feat2 = model1(data_VIS)
            detail1 = Encoder2(data_IR)
            detail2 = Encoder2(data_VIS)

            data_VIS = data_VIS[:, :1, :, :]
            output, _ = model2(data_VIS, feat1, feat2, detail1, detail2)

            fused = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            fused = (fused - fused.min()) / (fused.max() - fused.min()) * 255
            fused = np.squeeze(fused)
            result_img = Image.fromarray(fused.astype(np.uint8))
            result_img.save(os.path.join(test_out_folder, img_name))
        else:
            ir_img_rgb = image_read_cv2(os.path.join(ir_folder, img_name), mode='RGB')
            vis_img = image_read_cv2(os.path.join(vis_folder, img_name), mode='GRAY')

            ir_img_ycrcb = cv2.cvtColor(ir_img_rgb.astype('uint8'), cv2.COLOR_RGB2YCrCb)
            ir_img_y = ir_img_ycrcb[..., 0]
            ir_img_crcb = ir_img_ycrcb[..., 1:]  # shape: (H, W, 2)

            data_IR = prep_data(ir_img_y)
            data_VIS = prep_data(vis_img)

            feat1 = model1(data_IR)
            feat2 = model1(data_VIS)
            detail1 = Encoder2(data_IR)
            detail2 = Encoder2(data_VIS)

            data_VIS = data_VIS[:, :1, :, :]
            output, _ = model2(data_VIS, feat1, feat2, detail1, detail2)

            fused_y = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            fused_y = (fused_y - fused_y.min()) / (fused_y.max() - fused_y.min()) * 255
            fused_y = np.squeeze(fused_y).astype(np.uint8)

            fused_ycrcb = np.zeros_like(ir_img_ycrcb)
            fused_ycrcb[..., 0] = fused_y
            fused_ycrcb[..., 1:] = ir_img_crcb
            fused_rgb = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2RGB)

            result_img = Image.fromarray(fused_rgb)
            result_img.save(os.path.join(test_out_folder, img_name))

print(f"Output images saved to: {test_out_folder}")
