# 📄 Resume Classification Project (NLP & Machine Learning)

## 📌 Project Overview
This project focuses on building a Resume Classification system that predicts a candidate’s job role based on resume content.  
The model is trained on a limited set of predefined job categories such as **Peoplesoft**, **Workday**, **React Developer**, **SQL Developer**, and **Internship**.

The solution uses Natural Language Processing (NLP) techniques and a Machine Learning classification model, and it is deployed as an interactive **Streamlit web application**.

---

## 🎯 Problem Statement
Manual resume screening is time-consuming and inconsistent.  
This project aims to automate the initial screening process by classifying resumes into relevant job roles using textual analysis.

---

## 🧠 Machine Learning Approach
- **Text Cleaning & Preprocessing**
- **Feature Extraction:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Classification Algorithm:** Linear Support Vector Machine (LinearSVC)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Explainability:** Important keywords extracted from TF-IDF

---

## 🚀 Application Features
- Upload resumes in **PDF or DOCX** format
- Paste resume text manually
- Predict **job role with confidence score**
- Display **important keywords** influencing prediction
- Dark mode UI
- Download prediction result as **CSV or PDF**
- Analytics dashboard:
  - Total resumes processed
  - Role distribution
  - Confidence trend

---

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:**
  - scikit-learn
  - pandas, numpy
  - PyMuPDF
  - python-docx
  - Streamlit
- **Deployment:** Streamlit (Local)

---

## 📂 Project Structure
```

Resume_Classification_Project/
├── Resume_Classification_Project.py
├── resume_model.pkl
├── requirements.txt
├── Resumes.zip
└── README.md

````

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone [https://github.com/pankajbadipahadi/Resume_Classification.git]
cd Resume_Classification_Project
````

### 2️⃣ Create and activate environment

```bash
conda create -n resume-env python=3.9
conda activate resume-env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit application

```bash
streamlit run Resume_Classification_Project.py
```

---

## 📊 Sample Output

* **Predicted Role:** SQL Developer
* **Confidence:** 91.23%
* **Important Keywords:** sql, database, queries, data, analytics

---

## 🔮 Future Enhancements

* Add more job categories
* Improve dataset size for better generalization
* Use advanced NLP models (BERT)
* Deploy on cloud platforms
* Integrate with ATS systems

---

## 👤 Author

**Pankaj Raju Badipahadi**

---

⭐ If you find this project useful, feel free to give it a star!
