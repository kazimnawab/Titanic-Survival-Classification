# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
df = pd.read_csv(r"c:\Users\PC\Desktop\Titanic-Dataset.csv")   # Titanic dataset from Kaggle

# 3. Select Important Features
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

# 4. Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# 5. Convert Categorical Data to Numeric
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])   # male=1, female=0

# 6. Split Features and Target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Make Predictions
y_pred = model.predict(X_test)

# 10. Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
