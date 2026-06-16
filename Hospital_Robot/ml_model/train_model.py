#!/usr/bin/env python3
"""
=============================================================
HOSPITAL STOCK PREDICTION MODEL
=============================================================
WHY THIS MODEL?
- We use Random Forest because it handles mixed data types well
  (numbers like hour_of_day + categories like room_type, shift)
- It's robust to outliers (emergency cases can spike suddenly)
- Gives us feature importance so we know WHAT drives refill needs

WHAT IT PREDICTS:
- refill_needed: 1 = send robot to refill, 0 = stock is fine
- Threshold rule: if current_stock < 30 → almost always refill_needed=1
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

# ─────────────────────────────────────────────
# STEP 1: Generate a larger, realistic dataset
# ─────────────────────────────────────────────
# WHY: Your sample has only 6 rows. A model trained on 6 rows
# won't generalize. We'll generate 2000 rows that follow the
# same logic as your data so the model actually learns patterns.

np.random.seed(42)
n_samples = 2000

def generate_hospital_data(n):
    """
    Generate synthetic hospital stock data based on real patterns:
    - Night shifts (shift=1) have lower patient counts
    - ICU rooms (room_type=1) use stock faster
    - Emergency cases spike usage
    - Refill needed when stock < 30 OR usage is high relative to stock
    """
    data = []
    
    for _ in range(n):
        hour_of_day    = np.random.randint(0, 24)
        day_of_week    = np.random.randint(1, 8)        # 1=Mon ... 7=Sun
        room_type      = np.random.randint(0, 3)        # 0=General, 1=ICU, 2=Pharmacy
        
        # Shift logic: day=0 (6am-2pm), night=1 (2pm-10pm), graveyard=2 (10pm-6am)
        if 6 <= hour_of_day < 14:
            shift = 0
        elif 14 <= hour_of_day < 22:
            shift = 1
        else:
            shift = 2
        
        # Patient count depends on time of day
        if 8 <= hour_of_day <= 17:
            patient_count = np.random.randint(10, 30)
        else:
            patient_count = np.random.randint(2, 12)
        
        # Emergency cases - ICU has more
        if room_type == 1:  # ICU
            emergency_cases = np.random.randint(0, 5)
        else:
            emergency_cases = np.random.randint(0, 3)
        
        # Previous usage rate
        base_usage = patient_count * 0.3
        if room_type == 1:  # ICU uses more
            base_usage *= 1.5
        base_usage += emergency_cases * 1.2
        prev_usage = round(max(0.5, base_usage + np.random.normal(0, 0.5)), 1)
        
        # Current stock level
        current_stock = np.random.randint(1, 150)
        
        # REFILL LOGIC (the label we're predicting)
        # Refill needed if:
        # 1. Stock is below 30 (critical threshold)
        # 2. OR stock will run out before next restock (stock / usage < 6 hours)
        hours_until_empty = current_stock / max(prev_usage, 0.1)
        refill_needed = 1 if (current_stock < 30 or hours_until_empty < 6) else 0
        
        data.append([
            hour_of_day, patient_count, current_stock, shift,
            emergency_cases, prev_usage, day_of_week, room_type, refill_needed
        ])
    
    columns = [
        'hour_of_day', 'patient_count', 'current_stock', 'shift',
        'emergency_cases', 'prev_usage', 'day_of_week', 'room_type', 'refill_needed'
    ]
    return pd.DataFrame(data, columns=columns)

# ─────────────────────────────────────────────
# STEP 2: Add your original sample rows too
# ─────────────────────────────────────────────
sample_rows = [
    [6,  20, 14, 0, 1, 8.0, 1, 1, 1],
    [19,  7,  3, 1, 1, 2.2, 2, 1, 1],
    [14, 14, 92, 1, 2, 4.4, 3, 0, 0],
    [10, 18, 72, 0, 1, 8.8, 5, 0, 0],
    [7,  19, 26, 0, 2, 7.2, 2, 1, 1],
    [20,  8, 25, 1, 2, 2.5, 4, 1, 1],
]

columns = [
    'hour_of_day', 'patient_count', 'current_stock', 'shift',
    'emergency_cases', 'prev_usage', 'day_of_week', 'room_type', 'refill_needed'
]

df_original = pd.DataFrame(sample_rows, columns=columns)
df_generated = generate_hospital_data(n_samples)
df = pd.concat([df_original, df_generated], ignore_index=True)

print(f"[✓] Dataset created: {len(df)} rows")
print(f"    Refill needed:    {df['refill_needed'].sum()} ({df['refill_needed'].mean()*100:.1f}%)")
print(f"    No refill:        {(df['refill_needed']==0).sum()}")
print(f"\nSample data:\n{df.head(8).to_string()}")

# ─────────────────────────────────────────────
# STEP 3: Train/Test Split
# ─────────────────────────────────────────────
# WHY 80/20 split: 80% to learn patterns, 20% to check if it generalizes
X = df.drop('refill_needed', axis=1)
y = df['refill_needed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[✓] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 4: Train the Random Forest
# ─────────────────────────────────────────────
# WHY these parameters:
# - n_estimators=200: 200 trees → more stable predictions
# - max_depth=10: prevents overfitting (memorizing training data)
# - class_weight='balanced': handles unequal refill/no-refill counts

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # use all CPU cores
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# STEP 5: Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

# Cross-validation: 5-fold gives us a reliable accuracy estimate
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Test Accuracy:        {accuracy*100:.2f}%")
print(f"Cross-Val F1 (mean):  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Refill', 'Refill Needed']))

# Feature importance - tells us which inputs matter most
feature_importance = dict(zip(X.columns, model.feature_importances_))
print(f"\nFeature Importance (most → least):")
for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {feat:<20} {bar} {imp:.3f}")

# ─────────────────────────────────────────────
# STEP 6: Save model and metadata
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, 'stock_model.pkl')
joblib.dump(model, model_path)
print(f"\n[✓] Model saved → {model_path}")

# Save feature names and metadata for the dashboard
metadata = {
    "features": list(X.columns),
    "accuracy": round(accuracy * 100, 2),
    "cv_f1_mean": round(cv_scores.mean(), 3),
    "refill_threshold": 30,
    "feature_importance": {k: round(v, 4) for k, v in feature_importance.items()},
    "rooms": {
        "0": "General Ward",
        "1": "ICU",
        "2": "Pharmacy"
    }
}

meta_path = os.path.join(script_dir, 'model_metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[✓] Metadata saved → {meta_path}")

# ─────────────────────────────────────────────
# STEP 7: Quick prediction test
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"PREDICTION TESTS (using your original rows)")
print(f"{'='*50}")

test_cases = [
    {"hour_of_day":6,  "patient_count":20, "current_stock":14, "shift":0, "emergency_cases":1, "prev_usage":8.0, "day_of_week":1, "room_type":1, "expected":1},
    {"hour_of_day":14, "patient_count":14, "current_stock":92, "shift":1, "emergency_cases":2, "prev_usage":4.4, "day_of_week":3, "room_type":0, "expected":0},
    {"hour_of_day":7,  "patient_count":19, "current_stock":26, "shift":0, "emergency_cases":2, "prev_usage":7.2, "day_of_week":2, "room_type":1, "expected":1},
]

for i, tc in enumerate(test_cases):
    expected = tc.pop("expected")
    df_test  = pd.DataFrame([tc])  # DataFrame keeps feature names → no warning
    pred = model.predict(df_test)[0]
    prob = model.predict_proba(df_test)[0][1]
    status = "✓ CORRECT" if pred == expected else "✗ WRONG"
    room_names = {0:"General Ward", 1:"ICU", 2:"Pharmacy"}
    print(f"  Test {i+1}: Stock={tc['current_stock']}, Room={room_names[tc['room_type']]}")
    print(f"    Prediction: {'REFILL NEEDED' if pred==1 else 'Stock OK'} (confidence: {prob:.0%}) {status}")

print(f"\n[✓] Model training complete! Ready for dashboard and ROS2.")
