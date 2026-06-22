import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_STATE = 1
NUM_CLASSES = 4
BATCH_SIZE = 128
EPOCHS = 100

FIELDS_TO_DROP = [
    'sample', 'file_name', 'ML_1', 'ML_1_1', 'ML_1_1_1', 'ML_1_2', 'ML_2', 'ML_2_1_1',
    'ML_2_1_2', 'ML_2_1_3', 'ML_2_2', 'ML_2_3', 'ML_2_3_1', 'ML_2_4_1', 'ML_2_4_2',
    'ML_2_5_1', 'ML_2_6', 'ML_2_7_1', 'ML_2_7_2', 'ML_3', 'ML_3_1', 'ML_3_2', 'ML_3_3',
    'ML_3_4', 'ML_3_5', 'event_type'
]

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
labeled_dataset = pd.read_csv('data/labeled_processed_4Categories.csv')
labeled_dataset = labeled_dataset.fillna(labeled_dataset.median(numeric_only=True))

print(f"Dataset shape: {labeled_dataset.shape}")
print(f"\nClass distribution:\n{labeled_dataset['event_type'].value_counts()}")

# ─────────────────────────────────────────────
# 3. SPLIT: 70% train / 15% val / 15% test
# ─────────────────────────────────────────────
train, temp = train_test_split(
    labeled_dataset,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=labeled_dataset['event_type']
)
val, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp['event_type']
)

print(f"\nSplit sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")

train_features = train.drop(FIELDS_TO_DROP, axis=1)
train_labels   = train['event_type']

val_features   = val.drop(FIELDS_TO_DROP, axis=1)
val_labels     = val['event_type']

test_features  = test.drop(FIELDS_TO_DROP, axis=1)
test_labels    = test['event_type']

# ─────────────────────────────────────────────
# 4. SCALE FEATURES
# ─────────────────────────────────────────────
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)   # fit only on train!
val_features_scaled   = scaler.transform(val_features)
test_features_scaled  = scaler.transform(test_features)

# ─────────────────────────────────────────────
# 4b. RESHAPE FOR CNN: (samples, features) -> (samples, features, 1)
# ─────────────────────────────────────────────
train_features_cnn = train_features_scaled.reshape(train_features_scaled.shape[0], train_features_scaled.shape[1], 1)
val_features_cnn    = val_features_scaled.reshape(val_features_scaled.shape[0], val_features_scaled.shape[1], 1)
test_features_cnn   = test_features_scaled.reshape(test_features_scaled.shape[0], test_features_scaled.shape[1], 1)

print(f"\nCNN input shape: {train_features_cnn.shape}")

# ─────────────────────────────────────────────
# 5. ENCODE LABELS
# ─────────────────────────────────────────────
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded   = label_encoder.transform(val_labels)
test_labels_encoded  = label_encoder.transform(test_labels)

train_labels_onehot = keras.utils.to_categorical(train_labels_encoded, num_classes=NUM_CLASSES)
val_labels_onehot   = keras.utils.to_categorical(val_labels_encoded,   num_classes=NUM_CLASSES)
test_labels_onehot  = keras.utils.to_categorical(test_labels_encoded,  num_classes=NUM_CLASSES)

print(f"\nLabel classes: {list(label_encoder.classes_)}")

# ─────────────────────────────────────────────
# 6. CLASS WEIGHTS (handles imbalanced classes)
# ─────────────────────────────────────────────
class_weight_dict = {
    0: 1 / 0.367,   # No Event
    1: 1 / 0.081,   # Operational Switching
    2: 1 / 0.430,   # Abnormal Events
    3: 1 / 0.122,   # Fault Events
}

print("Class weights:")
for i, name in enumerate(['No Event', 'Operational Switching', 'Abnormal Events', 'Fault Events']):
    print(f"  Class {i} ({name}): {class_weight_dict[i]:.2f}")

# ─────────────────────────────────────────────
# 7. BUILD CNN MODEL
# ─────────────────────────────────────────────
n_features = train_features_cnn.shape[1]
print(f"\nNumber of input features: {n_features}")

model = keras.Sequential([
    layers.Input(shape=(n_features, 1)),

    # Conv block 1 — learns local patterns across neighboring features
    layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # Conv block 2
    layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # Conv block 3
    layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Flatten and classify
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# ─────────────────────────────────────────────
# 9. TRAIN
# ─────────────────────────────────────────────
history = model.fit(
    train_features_cnn,
    train_labels_onehot,
    validation_data=(val_features_cnn, val_labels_onehot),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# ─────────────────────────────────────────────
# 10. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(test_features_cnn, test_labels_onehot, verbose=0)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Per-class report
preds = np.argmax(model.predict(test_features_cnn), axis=1)
print("\nClassification Report:")

print(classification_report(
    test_labels_encoded,
    preds,
    target_names=[str(c) for c in label_encoder.classes_],
    zero_division=0
))

# ─────────────────────────────────────────────
# 11. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
cm = confusion_matrix(test_labels_encoded, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix — Test Set (CNN)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix_cnn.png")

# ─────────────────────────────────────────────
# 12. TRAINING CURVES PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'],     label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['accuracy'],     label='Train Acc')
ax2.plot(history.history['val_accuracy'], label='Val Acc')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('training_curves_cnn.png', dpi=150)
plt.show()
print("Training curves saved to training_curves_cnn.png")

# ─────────────────────────────────────────────
# 13. SAVE MODEL + ARTIFACTS
# ─────────────────────────────────────────────
model.save('oscillogram_classifier_cnn.keras')
joblib.dump(scaler,        'scaler_cnn.pkl')
joblib.dump(label_encoder, 'label_encoder_cnn.pkl')

print("\nSaved:")
print("  oscillogram_classifier_cnn.keras")
print("  scaler_cnn.pkl")
print("  label_encoder_cnn.pkl")