# Fake News Detection

This repository contains a single Jupyter notebook of task 3 in Elevvo Pathways' internship, which is building a **binary text classifier** to detect **Fake vs. Real** news articles using  NLP. It follows a clear, reproducible pipeline: data loading -> preprocessing -> TF-IDF feature -> linear model -> evaluation. 

**Dataset (Kaggle):** [Dataset Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---


## 1) **Data Loading**  
   - Loads `Fake.csv` and `True.csv` from the **Fake and Real News** dataset.  
   - Assigns binary labels and concatenates into a single dataframe.

## 2) **Text Preprocessing**  
   - Merges **Title + Content** into one text field.  
   - Cleans text (lowercasing, punctuation removal, whitespace normalization).  
   - Uses **spaCy** for tokenization, **stopword removal** (keeps negations like “not”), and **lemmatization**.

## 3) **Feature Engineering (TF-IDF)**  
   - Applies `TfidfVectorizer` on the cleaned text with **n-gram range = (1, 2)** (unigrams + bigrams).  

## 4) **Modeling**  
   - Trains a **Linear SVM (LinearSVC)** as the primary classifier.  

## 5) **Evaluation**  
   - Reports **Accuracy** and **F1-score** on the held-out test set, with Accuracy = 99.65% and F1-score = 99.64%.  
   - Prints a `classification_report` (precision/recall/F1 per class) to inspect any class imbalance.

> The code was written using Google Colab

