# 📧 Email Classifier: Spam vs Ham (NLP Project)

This project implements a machine learning-based email classifier that distinguishes **spam** emails from **ham** (non-spam) using **Natural Language Processing (NLP)** techniques. The solution demonstrates the full NLP workflow — from data preprocessing to model evaluation — and can be used as a foundation for production-grade spam detection systems.

---

## 🚀 Project Overview

Spam detection is one of the most common and practical NLP classification problems. This project uses a labeled dataset of emails and applies text cleaning, tokenization, feature extraction using **TF-IDF**, and supervised classification models to identify spam messages.

---

## 🧠 NLP Techniques Used

- **Text Cleaning**: Lowercasing, removing punctuation, and special characters.
- **Tokenization**: Splitting text into words.
- **Stopword Removal**: Eliminating common words that don't contribute to meaning.
- **Vectorization**: Using `CountVectorizer` and `TfidfVectorizer` to convert text into numerical format.
- **Model Training**: Using classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## 📂 Project Structure
📁 Email_Classifier_Spam_vs_Ham/
├── 📘 Email_Classifier_Spam_vs_Ham.ipynb # Main Jupyter Notebook
├── 📁 data/
│ └── spam_ham_dataset.csv # Raw labeled email dataset
├── 📄 README.md # Project documentation
├── 📄 requirements.txt # Dependencies (optional)


---

## 📊 Dataset

The dataset contains labeled email messages with the following structure:

| Label | Message                       |
|-------|-------------------------------|
| ham   | Hello, how are you?           |
| spam  | Win $1000 now by clicking... |

- Labels: `ham` (not spam) or `spam`
- Source: Public SMS/Email Spam Collection (e.g., UCI or Kaggle)

---

## 🛠️ How It Works

1. **Load and Explore Data**
   - Inspect class balance (spam vs ham)
   - Check for missing values

2. **Preprocessing Pipeline**
   - Normalize and clean the email text
   - Remove stopwords and punctuation
   - Tokenize and stem/lemmatize (optional)

3. **Feature Engineering**
   - Use `TfidfVectorizer` to convert text into numerical features

4. **Model Building**
   - Split the dataset (train/test)
   - Train using models like:
     - Multinomial Naive Bayes
     - Logistic Regression
     - Support Vector Machines (optional)

5. **Evaluation**
   - Evaluate using accuracy, precision, recall, and confusion matrix

6. **Prediction**
   - Test on custom emails to check spam prediction

---

## 🧪 Sample Results

| Model                | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| Naive Bayes         | 0.98     | 0.96      | 0.97   |
| Logistic Regression | 0.97     | 0.95      | 0.96   |

*(Actual values may vary based on data and preprocessing)*

---

## 🧰 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib / seaborn (optional for plots)


