# Simple Fraud Detection Demo
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and train model
url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Demo prediction
print("="*60)
print("FRAUD DETECTION SYSTEM - LIVE DEMO")
print("="*60)

# Test on a few samples
test_samples = X_test.iloc[:5]
predictions = model.predict(test_samples)
probabilities = model.predict_proba(test_samples)

for i, (idx, row) in enumerate(test_samples.iterrows()):
    actual = y_test.iloc[i]
    pred = predictions[i]
    prob = probabilities[i][1] * 100  # Fraud probability
    
    status = "🔴 FRAUD DETECTED" if pred == 1 else "✅ LEGITIMATE"
    actual_status = "FRAUD" if actual == 1 else "LEGITIMATE"
    
    print(f"\nTransaction {i+1}:")
    print(f"  Amount: ${row['Amount']:.2f}")
    print(f"  Prediction: {status}")
    print(f"  Fraud Probability: {prob:.2f}%")
    print(f"  Actual: {actual_status}")

print("\n" + "="*60)


