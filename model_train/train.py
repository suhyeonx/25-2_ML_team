import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from sklearn.preprocessing import RobustScaler
import joblib
from data_process_program.load_dataset import *
from data_process_program.augment_functions import *
from data_process_program.preprocess_functions import *

def build_model():
    model = models.Sequential([
        layers.InputLayer(shape=(250, 13)),
        layers.Conv1D(32, kernel_size=20, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=10, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            "accuracy",
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model

X_all, y_all = load_dataset("../data_process_program/all_dataset.csv")
X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_dataset(
    X_all, y_all, val_ratio=0.2, test_ratio=0.2
)

print(">> 데이터 증강 중...")
X_train_aug_raw, y_train_aug = augment_dataset(
    X_train_raw, y_train,
    normal_aug=4,
    falling_aug=16
)

print("[증강 결과]")
print(np.sum(y_train_aug == 0), np.sum(y_train_aug == 1))

X_train_feat = create_feature(X_train_aug_raw)
X_val_feat   = create_feature(X_val_raw)
X_test_feat  = create_feature(X_test_raw)

scaler = RobustScaler()

N_train, T, F = X_train_feat.shape
X_train_flat = X_train_feat.reshape(-1, F)
X_val_flat   = X_val_feat.reshape(-1, F)
X_test_flat  = X_test_feat.reshape(-1, F)

scaler.fit(X_train_flat)
joblib.dump(scaler, "robust_scaler.pkl")

X_train_final = scaler.transform(X_train_flat).reshape(N_train, T, F)
X_val_final   = scaler.transform(X_val_flat).reshape(X_val_feat.shape)
X_test_final  = scaler.transform(X_test_flat).reshape(X_test_feat.shape)

model = build_model()

early_stop = callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_final, y_train_aug,
    validation_data=(X_val_final, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save("best_fall_model.keras")
result = model.evaluate(X_test_final, y_test, verbose=0)
print(f"[TEST] loss: {result[0]:.4f} | auc: {result[1]:.4f} | acc: {result[2]:.4f} | recall: {result[3]:.4f}")

# 임계값 튜닝 부분
print(">> 예측임계값 튜닝")
model = tf.keras.models.load_model("best_fall_model.keras")

y_prob = model.predict(X_test_final, verbose=0).reshape(-1)

thresholds = np.arange(0.1, 1.0, 0.1)

print("Threshold | Recall | Precision | F1")

for thr in thresholds:
    y_pred = (y_prob > thr).astype(int)

    tp = np.sum((y_test == 1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"{thr:.1f} | {recall:.4f} | {precision:.4f} | {f1:.4f}")