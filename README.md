# 🧠 Fall Detection System using KFall Dataset

This repository contains a complete fall detection system using machine learning on wearable sensor data from the [KFall dataset](https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset?resource=download). It extracts features from accelerometer, gyroscope, and orientation data to detect fall events using a Support Vector Machine (SVM) classifier.

---

## 📥 Dataset Download

Due to size and licensing constraints, the KFall dataset is **not included** in this repository.

You can download it from Kaggle:

🔗 [KFall Dataset on Kaggle](https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset?resource=download)

### After downloading:
1. Extract the dataset.
2. Place it in your project directory like this:

```
KFall Dataset/
├── sensor_data/
│   ├── SA01/
│   ├── SA02/
│   └── ...
├── label_data/
│   ├── SA01_label.xlsx
│   ├── SA02_label.xlsx
│   └── ...
```

---

## 🧪 Machine Learning Model

- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Scaler**: StandardScaler
- **Class Balancing**: `class_weight='balanced'` to handle fall vs non-fall imbalance
- **Sliding Window**: 100 frames per window with 50% overlap

### 📊 Features Extracted (Per Window)
- Acceleration: Mean, Std, Max, Min of AccMag
- Gyroscope: Mean, Std, Max, Min of GyrMag
- Orientation: EulerX/Y/Z range
- Posture: Estimated Pitch, Roll
- Impact: Peak-to-peak AccMag
- Motion intensity: RMS of AccMag

---

## 🧾 Performance Metrics (Full Dataset)

| Metric           | Value    |
|------------------|----------|
| Accuracy         | 96.20%   |
| Precision (Fall) | 95.38%   |
| Recall (Fall)    | 96.93%   |
| Specificity      | 96.99%   |
| F1 Score (Fall)  | 96.15%   |

Confusion Matrix:
```
               Predicted
             |  0   |  1
         ----+------+-----
Actual   0   | 1422 |  67
         1   |  44  | 1388
```

---

## 🗂 Repository Structure

```
📦 KFall-Fall-Detection
├── KFall Dataset/                # Place extracted dataset here
├── full_kfall_analysis.py        # Full feature extraction + training pipeline
├── manual_test.py                # Manual testing script for individual trials
├── svm_fall_detection_model.joblib  # Trained SVM model
├── scaler_for_fall_detection.joblib # Fitted scaler
├── Fall_Detection_Project_Report.pdf # Project summary report
└── README.md
```

---

## 🧪 Manual Testing

Use `manual_test.py` to test the trained model on individual `.csv` files.

```bash
python manual_test.py
```

You’ll get:
- Per-window predictions
- Frame index of detected fall windows
- Overall conclusion (Fall or No Fall)

---

## 🔄 Planned Enhancements

- ✅ K-Fold Cross Validation (Stratified)
- 🔍 Hyperparameter Tuning with GridSearchCV
- 📈 Feature Expansion (Jerk, Entropy, FFT)
- 🌲 Try other classifiers (Random Forest, XGBoost, CNN)
- 📱 Real-time deployment on wearable devices

---

## 👨‍💻 Authors

- Dishant Kewat
- Diya Chimulwar
- Aastha Choudhari 
- Saurabh Wankhede 


---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [KFall Dataset on Kaggle](https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset?resource=download)
- PhysioNet & original KFall researchers
