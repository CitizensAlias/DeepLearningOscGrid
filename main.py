import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from FiveCats import five_cats_predict

# --- Load & prepare data ---
labeled_dataset = pd.read_csv('data/labeled_processed_validation.csv')

fields_to_drop = [
    'sample', 'file_name', 'ML_1', 'ML_1_1', 'ML_1_1_1', 'ML_1_2', 'ML_2', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3',
    'ML_2_2', 'ML_2_3', 'ML_2_3_1', 'ML_2_4_1', 'ML_2_4_2', 'ML_2_5_1', 'ML_2_6', 'ML_2_7_1', 'ML_2_7_2',
    'ML_3', 'ML_3_1', 'ML_3_2', 'ML_3_3', 'ML_3_4', 'ML_3_5', 'event_type'
]

#random_entries = labeled_dataset.sample(n=70000, random_state=42)
random_entries = labeled_dataset

y_true = random_entries['event_type'].tolist()
X = random_entries.drop(fields_to_drop, axis=1, errors='ignore')

# --- Predict (iterrows gives you a proper Series per row) ---
y_pred = [five_cats_predict(row.to_frame().T) for _, row in X.iterrows()]

# --- Evaluation ---
all_labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

# --- Confusion matrix heatmap ---
cm = confusion_matrix(y_true, y_pred, labels=all_labels)
plt.figure(figsize=(16, 13))
sns.heatmap(
    cm,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=all_labels, yticklabels=all_labels
)
plt.title("Five-Cats Confusion Matrix (n=1000)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("five_cats_confusion_matrix.png", dpi=150)
plt.show()