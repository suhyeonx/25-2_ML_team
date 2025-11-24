import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import joblib
from data_process_program.load_dataset import *
from data_process_program.augment_functions import  *
from data_process_program.preprocess_functions import *


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(20, 10, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(64, 12, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.1),

        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.1),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


# 메인 코드
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_PATH = "best_model.keras"

# 데이터셋 로드
print(">> 데이터 로드 중...")
X_all, y_all = load_dataset("../../data_process_program/all_dataset.csv")

X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_dataset(X_all, y_all, 0.2, 0.2)

# 데이터 증강
print(">> 데이터 증강 중...")
X_train_aug_raw, y_train_aug = augment_dataset(X_train_raw, y_train, normal_aug=4, falling_aug=16)

# feature 생성
print(">> feature 생성 중...")
X_train_feat = create_feature(X_train_aug_raw)
X_val_feat   = create_feature(X_val_raw)
X_test_feat  = create_feature(X_test_raw)

# RobustScaler 스케일러 적용
print(">> RobustScaler 적용 중...")
scaler = RobustScaler()

N_train, T, F = X_train_feat.shape
X_train_flat = X_train_feat.reshape(-1, F)
X_val_flat   = X_val_feat.reshape(-1, F)
X_test_flat  = X_test_feat.reshape(-1, F)

scaler.fit(X_train_flat)
joblib.dump(scaler, "robust_scaler.pkl")

X_train = scaler.transform(X_train_flat).reshape(N_train, T, F)
X_val   = scaler.transform(X_val_flat).reshape(X_val_feat.shape)
X_test  = scaler.transform(X_test_flat).reshape(X_test_feat.shape)

# 모델
input_shape = X_train.shape[1:]
model = build_model(input_shape)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max",
        patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc", mode="max",
        save_best_only=True
    )
]

# 학습
print(">> 모델 학습 시작...")
model.fit(
    X_train, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# 평가
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

print("[VAL]", f"loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f}")
print("[TEST]", f"loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")

y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
y_pred = (y_prob >= 0.4).astype(int)

print(classification_report(y_test, y_pred, digits=4))
print(confusion_matrix(y_test, y_pred))

print("\n베스트 모델 저장:", MODEL_PATH)