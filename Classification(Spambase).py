Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
# ================================
# SPAMBASE CLASSIFICATION MODELS
# CART, ID3, Random Forest, SVM
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("spambase.csv")

# Last column is target (spam or not)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# -------------------------------
# 2. Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Feature Scaling (for SVM)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Decision Tree - CART (Gini)
# -------------------------------
cart = DecisionTreeClassifier(criterion='gini', random_state=42)
cart.fit(X_train, y_train)

cart_pred = cart.predict(X_test)
cart_acc = accuracy_score(y_test, cart_pred)

# -------------------------------
# 5. Decision Tree - ID3 (Entropy)
# -------------------------------
id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3.fit(X_train, y_train)

id3_pred = id3.predict(X_test)
id3_acc = accuracy_score(y_test, id3_pred)

# -------------------------------
# 6. Random Forest
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -------------------------------
# 7. SVM
# -------------------------------
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)

svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)

# -------------------------------
# 8. Accuracy Comparison
# -------------------------------
print("Accuracy Scores")
print("-----------------------")
print("CART:", cart_acc)
print("ID3:", id3_acc)
print("Random Forest:", rf_acc)
print("SVM:", svm_acc)

# -------------------------------
# 9. Accuracy Bar Chart
# -------------------------------
models = ['CART', 'ID3', 'Random Forest', 'SVM']
scores = [cart_acc, id3_acc, rf_acc, svm_acc]

plt.figure(figsize=(8,5))
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

... # -------------------------------
... # 10. Confusion Matrix (Best Model)
... # -------------------------------
... # choose best model automatically
... best_model_pred = rf_pred
... best_model_name = "Random Forest"
... 
... cm = confusion_matrix(y_test, best_model_pred)
... 
... plt.figure(figsize=(5,4))
... sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
... plt.title(f"Confusion Matrix - {best_model_name}")
... plt.xlabel("Predicted")
... plt.ylabel("Actual")
... plt.show()
... 
... # -------------------------------
... # 11. Feature Importance (Random Forest)
... # -------------------------------
... importances = rf.feature_importances_
... indices = np.argsort(importances)[-10:]
... 
... plt.figure(figsize=(8,5))
... plt.barh(range(len(indices)), importances[indices])
... plt.yticks(range(len(indices)), X.columns[indices])
... plt.title("Top 10 Important Features")
... plt.show()
... 
... # -------------------------------
... # 12. Classification Report
... # -------------------------------
... print("\nClassification Report (Random Forest)")
