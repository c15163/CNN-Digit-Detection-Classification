# CNN-Digit-Detection-Classification

This project implements digit detection and classification using:

- Multi-scale MSER + Non-Max Suppression for region detection  
- A custom CNN, a simplified VGG-16, and a pretrained VGG-16 for digit classification  
- Focal loss to handle class imbalance  
- PyTorch and OpenCV  

---

## 📁 Files

- **`CNN_Detection_Classification.py`**  
  Full training, evaluation, and detection pipeline code.

- **`Final_project_report.pdf`**  
  Final project report describing methodology, experiments, and results.

- **`house_numbers/`**  
  Example real-world images used for detection + classification demo.

---

## 📥 Dataset Download (Google Drive)

This project uses a custom house number dataset created from real-world images.  
Due to file size limitations, the dataset is hosted externally on Google Drive.

🔗 **Download the full dataset (train + test)**  
https://drive.google.com/drive/folders/12rHH7h5AHNJru9kvpvem3GZs7vLdNxs9?usp=sharing

After downloading, place the files as follows:

```
project_root/
 ├── train/
 │    ├── train.zip
 │    ├── train_dataset_with_non_digit.npz
 ├── test/
 │    ├── test.zip
 │    ├── test_dataset_with_non_digit.npz
 └── house_numbers/
```

