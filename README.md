# 📰 News Article Classification & Topic Modeling

A Natural Language Processing (NLP) Project

---

## 📌 Overview

With the rapid growth of digital media, manually organizing news articles into categories is impractical. This project applies NLP techniques to **automatically classify news articles** and **uncover hidden thematic structures** within large text corpora.

In simple terms, the model can learn to tell **Politics**, **Sports**, **Business**, and **Technology** apart—without reading every article manually.

---

## 🎯 Objectives

- Automatically classify news articles into predefined categories
- Extract latent topics to understand underlying themes in the data
- Compare multiple machine learning models for text classification performance

---

## 📊 Dataset

The dataset consists of **thousands of news articles** across multiple domains. Each article includes:

- Raw textual content  
- A category label  

This makes it suitable for **both supervised classification** and **unsupervised topic modeling** tasks.

---

## 🧠 Methodology

### 🔹 Text Preprocessing
- Text normalization and cleaning
- Stopword removal
- Tokenization

### 🔹 Feature Engineering
- TF-IDF vectorization to capture term importance across documents

### 🔹 Model Training
- **Logistic Regression**  
- **Naive Bayes**  
- **Support Vector Machines (SVM)**  

### 🔹 Topic Modeling
- **Latent Dirichlet Allocation (LDA)** applied to identify hidden topics within the corpus
- Visualized topics with **word clouds** and **top words per topic**

---

## 📈 Evaluation Metrics

For classification models:

- Accuracy  
- Precision  
- Recall  
- F1-score  

For topic modeling:

- Topic coherence  
- Most frequent terms per topic  

---

## ✅ Results

- **TF-IDF** proved effective for classification  
- Classification models achieved **high and consistent accuracy**  
- Topic modeling revealed **coherent themes** aligned with real-world news domains  
- Visualizations such as **confusion matrices** and **word clouds** improved interpretability

---

## 🧩 Key Takeaways

- TF-IDF remains a **strong baseline** for NLP tasks  
- **Text preprocessing** significantly impacts model performance  
- Topic modeling adds **interpretability beyond labeled data**

---

## 🚀 Future Enhancements

- Integrate **word embeddings** such as Word2Vec or GloVe  
- Experiment with **transformer-based models** (BERT, RoBERTa)  
- Enhance **topic visualization** and coherence analysis  
- Deploy as a **real-time news classification app**

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **NLP Tools:** NLTK, SpaCy  
- **Visualization:** Matplotlib, Seaborn, WordCloud  

---

## 🏁 Conclusion

This project demonstrates an **end-to-end NLP workflow** — from raw text preprocessing to **supervised classification** and **unsupervised topic discovery**. It highlights how machine learning can extract **structure and insights** from large volumes of unstructured text in a **scalable** and **interpretable** manner.

---

## 📸 Visual Highlights (Optional for GitHub)

- Confusion matrix  
- Word clouds per topic  
- Accuracy comparison chart for models  

