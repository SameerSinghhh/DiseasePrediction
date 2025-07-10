# 🏥 Disease Prediction Model

> **A Machine Learning approach to predict diseases based on symptoms using advanced classification algorithms**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-latest-green.svg)](https://pandas.pydata.org/)

---

## 📊 Project Overview

This project implements a comprehensive machine learning pipeline for **disease classification** based on patient symptoms. Using a dataset of over 5,000 medical records, the model can predict potential diagnoses with high accuracy through advanced feature engineering and ensemble methods.

### ✨ Key Features

- 🔍 **Intelligent Symptom Analysis** - Processes 131+ unique symptoms
- 🧠 **Advanced ML Models** - Random Forest, Decision Tree & K-Nearest Neighbors classifiers
- 📈 **Dimensionality Reduction** - PCA optimization for better performance
- ⚙️ **Hyperparameter Tuning** - GridSearchCV for optimal model performance
- 📋 **Comprehensive Evaluation** - Detailed accuracy metrics and comparisons
- 🎯 **Clustering Analysis** - K-means clustering of disease patterns

---

## 🎯 Results Summary

We evaluated six models using balanced accuracy, precision, recall, and F1-score performance metrics:

- **🏆 PCA Random Forest** (Best Performer): 0.942, 0.946, 0.942, 0.943 respectively
- **PCA Decision Tree & PCA KNN**: Performed similarly well across all metrics  
- **Non-PCA Random Forest**: High 0.8x range performance
- **Non-PCA Decision Tree**: Lowest performance (0.6x range)
- **Non-PCA KNN**: Competitive performance

**Key Finding:** PCA significantly improved performance across all model types, with the PCA Random Forest achieving the highest scores. **We recommend the PCA Random Forest model** for its superior accuracy and ensemble method advantages in medical prediction scenarios.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Model
```bash
python DiseasePrediction/Disease_Prediction.py
```

---

## 🔬 Technical Approach

### 1. **Data Processing** 📋
- **Dataset**: 5,362 medical records with symptoms and diagnoses
- **Feature Engineering**: One-hot encoding of 131 unique symptoms
- **Data Split**: 80/20 train-test split with stratification

### 2. **Dimensionality Reduction** 📉
- **PCA Implementation**: Reduced to 40 components
- **Variance Explained**: ~99.9% of original variance retained
- **Performance**: Optimized feature space for faster training

### 3. **Machine Learning Models** 🤖

#### Random Forest Classifier
- **Hyperparameters**: Criterion, n_estimators, max_depth
- **Cross-Validation**: 6-fold CV for robust evaluation
- **Performance**: High accuracy with ensemble voting

#### Decision Tree Classifier
- **Hyperparameters**: Criterion, min_samples_split, max_depth
- **Optimization**: GridSearchCV for parameter tuning
- **Comparison**: Baseline model for ensemble validation

#### K-Nearest Neighbors (KNN)
- **Hyperparameters**: n_neighbors (tested k=1 to k=30)
- **Optimization**: Performance comparison across neighbor counts
- **Analysis**: Lazy learning algorithm with prediction-time trade-offs

### 4. **Clustering Analysis** 🎯
- **K-means Clustering**: Disease pattern identification
- **Cluster Optimization**: Silhouette analysis for optimal k=8
- **Visualization**: 3D PCA scatter plots with cluster assignments

---

## 📈 Results & Performance

| Model | Feature Space | Balanced Accuracy | Precision | Recall | F1-Score |
|-------|---------------|------------------|-----------|--------|----------|
| **Random Forest** | **PCA** | **0.942** | **0.946** | **0.942** | **0.943** |
| Decision Tree | PCA | High | High | High | High |
| KNN | PCA | High | High | High | High |
| Random Forest | Original | 0.8x range | 0.8x range | 0.8x range | 0.8x range |
| Decision Tree | Original | 0.6x range | 0.6x range | 0.6x range | 0.6x range |
| KNN | Original | Competitive | Competitive | Competitive | Competitive |

### 🎯 Key Insights
- **PCA Effectiveness**: Significantly improves model performance across all algorithms
- **Ensemble Advantage**: Random Forest outperforms single Decision Tree and KNN
- **Feature Importance**: Successfully identifies critical symptom patterns
- **Clustering Success**: Meaningful disease groupings with k=8 clusters

---

## 📁 Project Structure

```
DiseasePredictionModel/
├── DiseasePrediction/
│   ├── Disease_Prediction.py    # Main model implementation
│   └── diagnosis_and_symptoms.csv    # Dataset (add your data here)
└── README.md
```

---

## 🛠️ Technical Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
</p>

---

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DiseasePredictionModel.git
   cd DiseasePredictionModel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your dataset**
   - Place your `diagnosis_and_symptoms.csv` file in the `DiseasePrediction/` directory
   - Ensure the CSV format matches the expected structure

4. **Run the model**
   ```bash
   python DiseasePrediction/Disease_Prediction.py
   ```

---

## 📊 Model Workflow

```mermaid
graph TD
    A[📊 Load Dataset] --> B[🔄 Data Processing]
    B --> C[🏗️ Feature Engineering]
    C --> D[📉 PCA Reduction]
    D --> E[🔀 Train-Test Split]
    E --> F[🤖 Model Training]
    F --> G[⚙️ Hyperparameter Tuning]
    G --> H[📈 Model Evaluation]
    H --> I[🎯 Performance Comparison]
    I --> J[🔍 Clustering Analysis]
    J --> K[📋 Final Recommendations]
```

---

## 🎓 Academic Context

This project was developed as part of **ENGR 100** coursework, demonstrating:
- **Data Science Fundamentals**
- **Machine Learning Pipeline Design**
- **Statistical Analysis & Validation**
- **Healthcare Technology Applications**
- **Comparative Model Analysis**

**Team Members**: Sameer Singh, Joe Marcotte, Ian Nadeau, Sriram Kumaran

---

## 🔍 Future Enhancements

- [ ] **Deep Learning Models** - Neural network implementation
- [ ] **Real-time Prediction** - Web API for live diagnosis
- [ ] **Expanded Dataset** - Integration with larger medical databases
- [ ] **Feature Visualization** - Interactive symptom importance plots
- [ ] **Model Interpretability** - SHAP values for prediction explanations
- [ ] **Cross-validation Analysis** - More robust model validation
- [ ] **Ensemble Methods** - Combining multiple algorithms for better predictions

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

<p align="center">
  <strong>⭐ Star this repository if you found it helpful!</strong>
</p>

<p align="center">
  Made with ❤️ for advancing healthcare through machine learning
</p> 