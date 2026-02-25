# [FMC_UIA]-Deeper

## 🤝 Collaborating Institutions

- **ShanghaiTech University** (SHANGHAITECH), China
- **Beijing University of Technology** (BJUT), China
- **Dalian University of Technology** (DUT), China
- **Guangzhou National Laboratory** (GNL), China


### 1. Code structure

```

The train/ directory is NOT included in the repository.
You need to prepare and organize your own dataset following the structure below.

code/
├── train/
│   ├── csv_files/              # CSV index files
│   │   ├── Cls-Eight_1.Head_Position.csv
│   │   ├── Cls-Eight_2.Sacral_Position.csv
│   │   └── ...
│   ├── Classification/         # Classification task images
│   ├── Segmentation/          # Segmentation task images and masks
│   │   ├── Two/               # 2-class segmentation
│   │   ├── Three/             # 3-class segmentation
│   │   ├── Four/              # 4-class segmentation
│   │   └── Five/              # 5-class segmentation
│   ├── Detection/             # Detection task images
│   └── Regression/            # Regression task images
│       ├── fetal_femur.csv
│       └── ...
│
│── train.py
│── evaluate.py
│── model_factory.py
│── model.py
└── visualize.py

```

### 2. Train Model

```bash
git clone https://github.com/Daheilou/FMC_UIA_Deeper
python train.py
```

Training will automatically:
- Load data
- Train multi-task model
- Save best model as `best_model.pth`

---


### 3. [Experimental Results](https://www.codabench.org/competitions/11539/#/pages-tab)

| Phase | Segmentation_DSC | Segmentation_HD | Classification_Accuracy | Classification_AUC | Classification_F1_macro | Classification_F1_weighted | Classification_MCC | Detection_IoU | Regression_MRE |
|:-----:|:----------------:|:---------------:|:-----------------------:|:------------------:|:-----------------------:|:--------------------------:|:------------------:|:-------------:|:--------------:|
| val | 0.8928 | 30.057 | 0.8345 | 0.919 | 0.8213 | 0.8345 | 0.7217 | 0.7646 | 31.9822 |




### 4. Acknowledgements

This project benefits significantly from the open-source implementation of the **Foundation Model Challenge for Ultrasound Image Analysis (FM_UIA)**.

- **Challenge**: ISBI Foundation Model Challenge
- **Repository**: [github.com/lijiake2408/Foundation-Model-Challenge-for-Ultrasound-Image-Analysis](https://github.com/lijiake2408/Foundation-Model-Challenge-for-Ultrasound-Image-Analysis)

We thank the team for their contribution to the medical imaging community.

