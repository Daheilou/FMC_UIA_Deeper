# [FMC_UIA]-Deeper

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

train/ 

### 2. Train Model

```bash
python train.py
```

Training will automatically:
- Load data
- Train multi-task model
- Save best model as `best_model.pth`

---

Segmentation_HD	Classification_Accuracy	Classification_AUC	Classification_F1_macro	Classification_F1_weighted	Classification_MCC	Detection_IoU	Regression_MRE

0.8928	30.057	0.8345	0.919	0.8213	0.8345	0.7217	0.7646	31.9822
### 3. [Experimental Results](https://www.codabench.org/competitions/11539/#/pages-tab)

| Phase | Segmentation_DSC | Segmentation_HD | Classification_Accuracy | Classification_AUC | Classification_F1_macro | Classification_F1_weighted | Classification_MCC | Detection_IoU | Regression_MRE |
|:-----:|:----------------:|:---------------:|:-----------------------:|:------------------:|:-----------------------:|:--------------------------:|:------------------:|:-------------:|:--------------:|
| Validation | 0.8928 | 30.057 | 0.8345 | 0.919 | 0.8213 | 0.8345 | 0.7217 | 0.7646 | 31.9822 |


