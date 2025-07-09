# 💳 Credit Card Fraud Detection

This project presents a deep learning approach for detecting fraudulent credit card transactions using fully connected neural networks. It leverages a **pipeline that integrates SMOTE + Tomek Links** to automatically address class imbalance during model training.

Multiple neural network configurations were tested by varying **activation functions** (`ReLU`, `Tanh`) and **optimizers** (`SGD`, `Adam`, `RMSprop`). The **best-performing model was selected** based on metrics such as **F1-score**, **ROC-AUC**, and **Confusion Matrix** performance.

---

## 📁 Project Structure
- `Fraud_Detection.ipynb`: Complete notebook with preprocessing, model training, evaluation, and visualization.

---

## 🚀 Features
- 🧠 Built with fully connected **neural networks** using Keras/TensorFlow  
- ⚖️ Class imbalance handled using **SMOTE + Tomek Links** in a pipeline  
- 🔧 Tested with multiple combinations of **activation functions** and **optimizers**  
- 🏆 **Best model selected** based on ROC-AUC and F1-score  
- 📊 Evaluated using **Confusion Matrix**, **Classification Report**, and ROC Curve  
- 📈 Visualized training loss, accuracy, and performance comparisons  

---

## 📦 Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn, Imbalanced-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## 🧪 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
Install dependencies:
pip install -r requirements.txt

Run the notebook:
jupyter notebook Fraud_Detection.ipynb

✅ Results
Tested neural networks with combinations of optimizers and activation functions

Best model selected based on F1-score and ROC-AUC

Class imbalance handled effectively using pipeline-based SMOTE + Tomek

Achieved high precision and recall for fraud cases

