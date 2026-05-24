# 🚢 Titanic Survival Classification Using Machine Learning

A Python-based Machine Learning project that predicts Titanic passenger survival using a Random Forest Classifier. Implements data cleaning, mean imputation, label encoding, and 80/20 train-test split. Achieves ~80% validation accuracy using Pclass, Sex, Age, and Fare as key features.

---

## 👨‍💻 Student Information

| Field | Details |
|-------|---------|
| Name | Kazim Nawab |
| Domain | Data Science |
| Department | Computer Science |
| Email | Kazimnawabgd@gmail.com |
| Contact | 03179542997 |
| Company | Arch Technologies |
| Submission Date | May 23, 2026 |

---

## 🛠️ Technologies Used

- Python 3.x
- pandas
- scikit-learn
- LabelEncoder
- RandomForestClassifier
- SimpleImputer / fillna

---

## 📁 Dataset

- **Source:** Kaggle Titanic Dataset
- **Target Column:** `Survived` (0 = Deceased, 1 = Survived)

### Features Used:
| Column | Description |
|--------|-------------|
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Sex | Gender (male/female) |
| Age | Passenger age |
| Fare | Ticket price |

---

## ▶️ How to Run

**Step 1 — Install Libraries:**
```bash
pip install pandas scikit-learn
```

**Step 2 — Place Dataset:**
```
C:\Users\PC\Desktop\Titanic-Dataset.csv
```

**Step 3 — Run Script:**
```bash
python kazim_nawab_Data_science_month_1.py
```

---

## 📊 Model Results

- ✅ **Accuracy: 79.89% (~80%)**
- True Negatives: 89
- True Positives: 54
- False Positives: 16
- False Negatives: 20

### Classification Report:
| Class | Precision | Recall |
|-------|-----------|--------|
| Deceased (0) | 82% | 85% |
| Survived (1) | 77% | 73% |

---

## 🔄 Project Workflow

1. Load CSV data using pandas
2. Select important columns
3. Fill missing values with mean
4. Convert Sex column to numbers
5. Split into X (features) and y (target)
6. Train/Test split (80/20)
7. Train RandomForestClassifier
8. Predict and evaluate results

---

## 📝 Conclusion

The Random Forest model achieved ~80% accuracy. Most important features were **Sex** and **Pclass**, matching historical Titanic evacuation priorities.
