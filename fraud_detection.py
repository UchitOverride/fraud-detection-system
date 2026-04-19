# ===== FRAUD DETECTION SYSTEM =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("="*60)
print("CREDIT CARD FRAUD DETECTION SYSTEM")
print("="*60)

# 1. LOAD DATA
print("\n[1/5] Loading dataset...")
url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)
print(f"✓ Dataset loaded: {df.shape[0]} transactions, {df.shape[1]} features")
print(f"✓ Fraudulent: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")

# 2. DATA PREPROCESSING
print("\n[2/5] Preprocessing data...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# 3. BUILD MODELS
print("\n[3/5] Training machine learning models...")
print("-"*60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...", end=" ")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    results.append({
        'Model': name,
        'Accuracy (%)': round(acc*100, 2),
        'Precision (%)': round(prec*100, 2),
        'Recall (%)': round(rec*100, 2),
        'F1-Score (%)': round(f1*100, 2)
    })
    
    print(f"✓ Done! Accuracy: {acc*100:.2f}%")
    
    if f1 > best_score:
        best_score = f1
        best_model = (name, model, pred)

# 4. DISPLAY RESULTS
print("\n[4/5] Model Comparison Results:")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 5. VISUALIZATIONS
print("\n[5/5] Generating visualizations...")

# Confusion Matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cm = confusion_matrix(y_test, best_model[2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix - {best_model[0]}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Model comparison chart
fig, ax = plt.subplots(figsize=(10, 6))
results_df.set_index('Model')[['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']].plot(kind='bar', ax=ax)
ax.set_title('Model Performance Comparison')
ax.set_ylabel('Score (%)')
ax.set_xlabel('Models')
ax.legend(loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✅ PROJECT COMPLETE!")
print(f"✅ Best Model: {best_model[0]} (F1-Score: {best_score*100:.2f}%)")
print("="*60)