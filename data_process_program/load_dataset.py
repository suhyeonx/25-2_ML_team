import pandas as pd
import numpy as np


# 0번 라벨(normal) + 2번 라벨(fake) -> 0번 라벨로 바꿔서 0과 1라벨로 이루어진 데이터셋 반환
def load_dataset(file="./all_dataset.csv"):
    df = pd.read_csv(file)
    df["label"] = df["label"].apply(lambda x: 1 if x == 1 else 0)

    df = df.sample(frac=1).reset_index(drop=True)

    feature_cols = [col for col in df.columns if col.startswith("v")]

    X_all = df[feature_cols].values
    y_all = df["label"].values

    return X_all, y_all

# 데이터 셋을 train, val, test로 분리하는 함수. val과 test는 normall:falling 비율이 1:1이 되도록함\
def split_dataset(X_all, y_all, val_ratio=0.2, test_ratio=0.2):
    y_all = np.asarray(y_all)

    idx_normal = np.where(y_all == 0)[0]
    idx_falling = np.where(y_all == 1)[0]

    # 셔플
    np.random.shuffle(idx_normal)
    np.random.shuffle(idx_falling)

    total_len = len(y_all)
    val_each = int(total_len * val_ratio) // 2
    test_each = int(total_len * test_ratio) // 2

    # val 데이터셋
    val_idx = np.concatenate([idx_normal[:val_each], idx_falling[:val_each]])

    # test 데이터셋
    test_idx = np.concatenate([idx_normal[val_each:val_each + test_each], idx_falling[val_each:val_each + test_each]])

    # train 데이터셋
    train_idx = np.concatenate([idx_normal[val_each + test_each:], idx_falling[val_each + test_each:]])

    # 데이터셋 분리후 내부에서 한번 더 셔플
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    # 데이터셋 반환
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]

    X_val = X_all[val_idx]
    y_val = y_all[val_idx]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    print(f"[Split 완료] Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
    print(f" > Val normal/falling : {np.sum(y_val == 0)} / {np.sum(y_val == 1)}")
    print(f" > Test normal/falling : {np.sum(y_test == 0)} / {np.sum(y_test == 1)}")

    return X_train, y_train, X_val, y_val, X_test, y_test