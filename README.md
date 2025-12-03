
# ATDFusion: Adapter-Tuned Dual-Branch Network for Multimodal Medical Image Fusion

This is official Pytorch implementation of **"ATDFusion: Adapter-Tuned Dual-Branch Network for Multimodal Medical Image Fusion"**  

---

## 📢 News

* **[2025.12]** To demonstrate the **generalization capability** of ATDFusion, we conducted additional experiments on the **HECKTOR 2025** dataset. The pre-processed testing data (sliced from NIfTI to PNG format) is now available in this repository.
    * *Note: For the full raw dataset, please apply via the [official HECKTOR 2025 website](https://hecktor25.grand-challenge.org/).*

---

##  Requirements

- Python >= 3.11  
- PyTorch 2.4.0  
- torchvision 0.19.0  
- numpy 2.2.1  
- opencv-python 4.10.0  
- Pillow 11.1.0  
- tqdm 4.67.1  

Install dependencies:
```bash
pip install -r requirements.txt
```

---

##  Dataset Preparation

Please download paired medical images from the [Harvard Medical School website](https://www.med.harvard.edu/AANLIB/home.html).
Organize the dataset into the following structure under `Medical_dataset_test/`:

```
Medical_dataset_test/
├── MRI-CT/
│   ├── MRI/
│   └── CT/
├── MRI-PET/
│   ├── MRI/
│   └── PET/
└── MRI-SPECT/
    ├── MRI/
    └── SPECT/
```

---

##  Usage

1. Download pretrained model weights from: **\[[link to be provided](https://pan.baidu.com/s/1kCttUIk-AOzc_IMlUcWM_g?pwd=wr5a)]**
   Place the model file in the `checkpoints/` directory.

2. Run inference:

```bash
python test.py
```

3. Results will be saved in the `test_result/` folder.

---

## Citation

```bibtex
@misc{li2025atdfusion,
  title={ATDFusion: Adapter-Tuned Dual-Branch Network for Medical Image Fusion},
  author={Chenyang Li and Rui Zhu and Hang Zhao and Xiongfei Li and Xiaoli Zhang},
  year={2025},
}
```
