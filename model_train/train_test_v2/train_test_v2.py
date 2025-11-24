import librosa
import pandas as pd
import numpy as np
import random as pyrandom
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import *
from data_process_program.load_dataset import *
from data_process_program.augment_functions import *
from data_process_program.preprocess_functions import *


# 설정
# 데이터셋 파일
FILE_PATH = "../../data_process_program/all_dataset.csv"

# 증강 배수 설정
NORMAL_AUG = 4
FALLING_AUG = 6

# 학습 관련 설정
EARLY_STOPPING_PATIENCE = 50
EPOCHS= 50
BATCH_SIZE= 32
LEARNING_RATE = 0.0001

# 기타 하이퍼파라미터
THRESHOLD = 0.5  # 예측시 확률 임계값


# 모델
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

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])

    return model

# val 데이터용 데이터셋 분리함수
def split_balanced_val(X, y, val_ratio=0.1):
    idx_normal = np.where(y == 0)[0]
    idx_falling = np.where(y == 1)[0]

    np.random.shuffle(idx_normal)
    np.random.shuffle(idx_falling)

    val_len = int(len(y) * val_ratio)

    n = val_len // 2

    val_idx = np.concatenate((idx_normal[:n], idx_falling[:n]))
    train_idx = np.concatenate((idx_normal[n:], idx_falling[n:]))

    np.random.shuffle(val_idx)
    np.random.shuffle(train_idx)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# 스케일러 함수
def scale(X_train, X_val, X_test):
    scaler = RobustScaler()

    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape

    X_train_flat = X_train.reshape(-1, 13)
    X_val_flat = X_val.reshape(-1, 13)
    X_test_flat = X_test.reshape(-1, 13)

    scaler.fit(X_train_flat)

    X_train_scaled = scaler.transform(X_train_flat).reshape(train_shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(val_shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(test_shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# 메인 코드
# 데이터셋 로드
X_all, y_all = load_dataset(FILE_PATH)

# 5겹 K-폴드 준비
k_fold = StratifiedKFold(n_splits=5, shuffle=True)

# 결과 저장용
history_results = []
metrics_results = {"acc": [], "auc": [], "recall": [], "f1": []}
cm_results = np.zeros((2, 2))  # 혼동행렬

# K-Fold 시작
print("\n5-Fold 검증 시작...")

for fold, (train_idx, test_idx) in tqdm(enumerate(k_fold.split(X_all, y_all))):
    print(f"\n========== Fold {fold + 1} / 5 ==========")

    # 데이터셋 분할
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    X_train, y_train, X_val, y_val = split_balanced_val(X_train, y_train, 0.1)
    print(f"  - Split 결과:")
    print(f"    > Train: {len(y_train)}")
    print(f"    > Val: {len(y_val)}")
    print(f"    > Test: {len(y_test)}")

    # Train 데이터 증강
    print("  - Train 데이터 증강 중...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, NORMAL_AUG, FALLING_AUG)

    # 데이터 전처리 및 스케일링
    X_train_feature = create_feature(X_train_aug)
    X_val_feature = create_feature(X_val)
    X_test_feature = create_feature(X_test)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale(X_train_feature, X_val_feature, X_test_feature)

    print("  - 모델 학습 시작...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', patience=EARLY_STOPPING_PATIENCE,
                                                  restore_best_weights=True, verbose=0)
    model = build_model()
    history = model.fit(
        X_train_scaled, y_train_aug,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )

    # 모델 평가
    history_results.append(history.history)

    y_pred_prob = model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_results["acc"].append(acc)
    metrics_results["auc"].append(auc)
    metrics_results["recall"].append(recall)
    metrics_results["f1"].append(f1)
    cm_results += confusion_matrix(y_test, y_pred)

    print(f"  >> Fold {fold + 1}: ACC={acc:.4f}, AUC={auc:.4f}\n")

# 데이터 시각화 부분은 AI의 도움을 받았습니다
min_epochs = min([len(h['loss']) for h in history_results])
epochs_range = range(1, min_epochs + 1)

avg_loss = np.mean([h['loss'][:min_epochs] for h in history_results], axis=0)
avg_val_loss = np.mean([h['val_loss'][:min_epochs] for h in history_results], axis=0)
avg_acc = np.mean([h['accuracy'][:min_epochs] for h in history_results], axis=0)
avg_val_acc = np.mean([h['val_accuracy'][:min_epochs] for h in history_results], axis=0)
avg_auc = np.mean([h['AUC'][:min_epochs] for h in history_results], axis=0)
avg_val_auc = np.mean([h['val_AUC'][:min_epochs] for h in history_results], axis=0)

avg_cm = cm_results / 5.0
labels = ['Accuracy', 'AUC', 'Recall', 'F1-Score']
means = [np.mean(metrics_results[k]) for k in ['acc', 'auc', 'recall', 'f1']]
stds = [np.std(metrics_results[k]) for k in ['acc', 'auc', 'recall', 'f1']]

# 2. ★ 화면 레이아웃 잡기 (여기가 핵심!) ★
fig = plt.figure(figsize=(18, 12))  # 전체 창 크기 넉넉하게
gs = fig.add_gridspec(2, 6)  # 2행 6열로 바둑판 쪼개기

# --- [상단 1행] 3개 그래프 (각각 2칸씩 차지) ---
ax1 = fig.add_subplot(gs[0, 0:2])  # 0~2칸
ax1.plot(epochs_range, avg_loss, 'b-', label='Train')
ax1.plot(epochs_range, avg_val_loss, 'r--', label='Val')
ax1.set_title('Average Loss')
ax1.legend();
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2:4])  # 2~4칸
ax2.plot(epochs_range, avg_acc, 'b-', label='Train')
ax2.plot(epochs_range, avg_val_acc, 'r--', label='Val')
ax2.set_title('Average Accuracy')
ax2.legend();
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 4:6])  # 4~6칸
ax3.plot(epochs_range, avg_auc, 'b-', label='Train')
ax3.plot(epochs_range, avg_val_auc, 'r--', label='Val')
ax3.set_title('Average AUC')
ax3.legend();
ax3.grid(True, alpha=0.3)

# --- [하단 2행] 2개 그래프 (각각 3칸씩 차지 -> 큼직하게!) ---
# 1. 혼동 행렬 (왼쪽 절반)
ax4 = fig.add_subplot(gs[1, 0:3])  # 0~3칸
sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Purples', ax=ax4, annot_kws={"size": 14},  # 글자도 좀 키움
            xticklabels=['Pred Normal', 'Pred Fall'],
            yticklabels=['Act Normal', 'Act Fall'])
ax4.set_title('Average Confusion Matrix', fontsize=14)
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')

# 2. 성능 지표 막대 (오른쪽 절반)
ax5 = fig.add_subplot(gs[1, 3:6])  # 3~6칸
bars = ax5.bar(labels, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax5.set_title('Final Performance Metrics', fontsize=14)
ax5.set_ylim(0, 1.15)
ax5.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()