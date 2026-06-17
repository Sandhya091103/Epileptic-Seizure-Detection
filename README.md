# Epileptic Seizure Detection Using Deep Learning

A deep learning project for EEG-based epileptic seizure detection using 1D Convolutional Neural Networks (CNNs).

## Project Goals
- Achieve **92% classification accuracy**
- Reduce false alarms by **30%**
- Evaluate using Confusion Matrix and ROC Curve

## Dataset
[UCI Epileptic Seizure Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
- 11,500 EEG samples
- 178 time-series features per sample
- 5 classes (Class 1 = seizure activity, Classes 2–5 = non-seizure)

## Tech Stack
- Python, TensorFlow/Keras
- NumPy, Pandas, SciPy
- scikit-learn, Matplotlib, Seaborn

## Project Structure
```
Epileptic_Seizure_Detection/
├── data/                          ← place data.csv here (not tracked by git)
├── seizure_detection.ipynb        ← main notebook
├── model/
│   └── seizure_model.h5           ← saved trained model
├── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

Place the dataset CSV in the `data/` folder, then open and run `seizure_detection.ipynb`.

## Author
**Sandhya Singh**
- GitHub: [@Sandhya091103](https://github.com/Sandhya091103)
- Email: singhsandhya171@gmail.com
