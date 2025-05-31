#  Deep Learning Model untuk Prediksi Diabetes

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

##  Deskripsi Proyek

Proyek ini mengembangkan model deep learning untuk memprediksi kemungkinan diabetes berdasarkan data medis pasien. Model menggunakan neural network dengan arsitektur yang dioptimalkan, teknik regularisasi, dan berbagai metode peningkatan performa untuk mencapai akurasi sederhana dalam klasifikasi biner.

###  Tujuan Utama
- Mengembangkan model dasar prediksi diabetes yang akurat dan reliabel 
- Mengimplementasikan best practices dalam deep learning untyk masalah medis
- Menyediakan tools yang mudah digunakan untuk prediksi pada data baru
- Memberikan insight tentang faktor faktor yang mempengaruhi diabetes

##  Dataset

Dataset yang digunakan adalah **Pima Indians Diabetes Database** yang berisi data medis dari 768 wanita dengan usia minimal 21 tahun dari suku Pima Indian.

### Fitur-fitur Dataset:
| Fitur | Deskripsi | Satuan |
|-------|-----------|--------|
| `Pregnancies` | Jumlah kehamilan | - |
| `Glucose` | Kadar glukosa darah (2 jam post-glucose tolerance test) | mg/dL |
| `BloodPressure` | Tekanan darah diastolik | mmHg |
| `SkinThickness` | Ketebalan lipatan kulit triceps | mm |
| `Insulin` | Kadar insulin serum (2 jam post-test) | μU/mL |
| `BMI` | Body Mass Index | kg/m² |
| `DiabetesPedigreeFunction` | Fungsi riwayat diabetes dalam keluarga | - |
| `Age` | Usia | tahun |
| `Outcome` | Target variable (0: Non-diabetes, 1: Diabetes) | - |

### Karakteristik Dataset:
- **Total sampel**: 768 data
- **Fitur**: 8 fitur numerik
- **Target**: Binary classification (0/1)
- **Distribusi kelas**: ~65% Non-diabetes, ~35% Diabetes

##  Fitur Utama

###  Preprocessing Tambahan
- **Penanganan missing values**: Mengganti nilai 0 yang tidak valid dengan median
- **Feature scaling**: StandardScaler untuk normalisasi fitur
- **Data visualization**: Analisis distribusi dan korelasi fitur

###  Arsitektur Model 
- **Deep Neural Network**: 3 hidden layers (64-32-16 neurons)
- **Regularization**: L1 & L2 regularization untuk mencegah overfitting
- **Batch Normalization**: Stabilitas training dan faster convergence
- **Dropout**: Dropout layers dengan rate 0.2-0.3

###  Training Enhancement
- **Early Stopping**: Automatic training termination untuk optimal performance
- **Model Checkpointing**: Menyimpan model terbaik selama training
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **K-Fold Cross Validation**: Robust model evaluation

###  Comprehensive Analysis
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Training curves, Confusion Matrix, ROC Curve
- **Feature Importance**: Analisis kontribusi setiap fitur
- **Model Comparison**: Perbandingan dengan baseline model

## 🛠️ Instalasi dan Setup

### Prerequisites
- Python 3.7 atau lebih tinggi
- Git (untuk cloning repository)

### 1. Clone Repository
```bash
git clone https://github.com/rexzea/diabetes-prediction-deep-learning.git
cd diabetes-prediction-deep-learning
```

### 2. Buat Virtual Environment
```bash
# Menggunakan venv
python -m venv diabetes_env
source diabetes_env/bin/activate  # Linux/Mac
# atau
diabetes_env\Scripts\activate     # Windows

# Menggunakan conda
conda create -n diabetes_env python=3.8
conda activate diabetes_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
joblib>=1.0.0
jupyter>=1.0.0
```


##  Cara Penggunaan

### 1. Training Model dari Awal

```python
# Import library
from src.data_preprocessing import preprocess_data
from src.model_builder import create_improved_model
from src.training import train_model

# Load dan preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data('data/diabetes.csv')

# Buat model
model = create_improved_model(input_dim=X_train.shape[1])

# Training model
history = train_model(model, X_train, y_train, X_test, y_test)
```

### 2. Menggunakan Model yang Sudah Dilatih

```python
from src.prediction import DiabetesPredictor

# Inisialisasi predictor
predictor = DiabetesPredictor('models/diabetes_prediction_model.h5', 
                             'models/diabetes_scaler.pkl')

# Prediksi untuk satu pasien
patient_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
result = predictor.predict_single(patient_data)
print(f"Prediksi: {result['prediction']}")
print(f"Probabilitas: {result['probability']:.2f}%")

# Prediksi untuk multiple pasien dari CSV
results = predictor.predict_from_csv('data/new_patients.csv')  # Contoh 
print(results.head())
```

### 3. Menjalankan Jupyter Notebook

```bash
jupyter notebook notebooks/02_model_development.ipynb
```

##  Hasil dan Performa Model

### Model Performance
| Metric | Baseline Model | Improved Model |
|--------|----------------|----------------|
| **Accuracy** | 76.62% | **84.42%** |
| **Precision** | 0.73 | **0.82** |
| **Recall** | 0.68 | **0.79** |
| **F1-Score** | 0.70 | **0.80** |
| **AUC-ROC** | 0.81 | **0.89** |

### K-Fold Cross Validation
- **Mean Accuracy**: 83.24% ± 2.15%
- **Consistency**: High stability across folds

### Feature Importance
1. **Glucose** (28.45%) - Faktor paling berpengaruh
2. **BMI** (18.32%) - Indeks massa tubuh
3. **Age** (15.67%) - Usia pasien
4. **DiabetesPedigreeFunction** (12.89%) - Riwayat keluarga
5. **Pregnancies** (8.91%) - Jumlah kehamilan

##  Analisis Model

### Kelebihan Model
-  **Akurasi tinggi**: 84.42% pada test set
-  **Robust**: Konsisten di berbagai fold validation
-  **Interpretable**: Analisis feature importance yang jelas
-  **Production-ready**: Model dapat disimpan dan dimuat dengan mudah

### Limitasi
-  **Dataset bias**: Terbatas pada populasi Pima Indian women
-  **Feature terbatas**: Hanya 8 fitur medis
-  **Imbalanced data**: Distribusi kelas tidak seimbang (65:35)

### Rekomendasi Pengembangan
-  **Data augmentation**: Menambah variasi data
-  **Feature engineering**: Membuat fitur baru dari kombinasi existing features
-  **Ensemble methods**: Kombinasi multiple models
-  **Hyperparameter tuning**: Optimasi lebih lanjut dengan Grid/Random Search

##  Testing

Jalankan unit tests untuk memastikan :

```bash
# Jalankan semua tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_model.py -v

# Test dengan coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

##  Dokumentasi API

### DiabetesPredictor Class

```python
class DiabetesPredictor:
    """
    kelas untuk melakukan prediksi diabetes menggunakan trained model.
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Init predictor model dan scaler path.
        
        Args:
            model_path: Path ke saved model (.h5)
            scaler_path: Path ke saved scaler (.pkl)
        """
    
    def predict_single(self, patient_data: list) -> dict:
        """
        Prediksi untuk satu pasien.
        
        Args:
            patient_data: list berisi 8 fitur medis
            
        Returns:
            dict: {'prediction': str, 'probability': float}
        """
    
    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        prediksi untuk multiple pasien dari CSV file.
        
        Args:
            csv_path: Path ke CSV file
            
        Returns:
            pd.DataFrame: DataFrame dengan hasil prediksi
        """
```

##  Contributing

Kami menyambut kontribusi dari komunitas! Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk guidelines.

### Cara Berkontribusi:
1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

### Area yang Butuh Kontribusi:
-  **New features**: Implementasi algoritma ML lain
-  **Bug fixes**: Perbaikan bug yang ditemukan
-  **Documentation**: Perbaikan dan penambahan dokumentasi
-  **Testing**: Penambahan test cases
-  **UI/UX**: Pengembangan web interface

##  License

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

```
MIT License

Copyright (c) 2024 Diabetes Prediction Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

##  Acknowledgments

- **Dataset**: Pima Indians Diabetes Database dari UCI Machine Learning Repository
- **Inspiration**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Libraries**: TensorFlow, Scikit-learn, Pandas, NumPy teams
- **Community**: Kaggle dan Stack Overflow communities

##  Contact

**Maintainer**: rexzea
-  Email: futzfary@gmail.com
-  GitHub: [@rexzea]([https://github.com/yourusername](https://github.com/rexzea?fbclid=PAQ0xDSwKnNYBleHRuA2FlbQIxMQABpzZj-DhcoKT7nJAlQajpUiZYp12dgNSQBjbwLm6WEuqEizn5dMk8MFLGsF2u_aem_TvFDhNFn8BEO-3EyK1iH0A ))


**Project Link**: [https://github.com/rexzea/diabetes-prediction-deep-learning](https://github.com/rexzea/diabetes-prediction-deep-learning)

---

##  Disclaimer

 **PENTING**: Model ini dikembangkan untuk tujuan **edukasi dan research**. Hasil prediksi **TIDAK boleh** digunakan sebagai pengganti diagnosis medis profesional. Selalu konsultasikan dengan dokter atau tenaga medis yang kompeten untuk diagnosis dan pengobatan diabetes.

---

<div align="center">

**⭐ Jika proyek ini bermanfaat, berikan star di GitHub! ⭐**

Made with ❤️ for the medical AI community

[⬆ Back to Top](#-deep-learning-model-untuk-prediksi-diabetes)

</div>
