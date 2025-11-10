import pandas as pd


def labeling(origin_label):
    return 1 if origin_label == 1 else 0

file_path = "all_dataset.csv"

df = pd.read_csv(file_path)
df['binary_label'] = df['label'].apply(labeling)


feature_cols = [col for col in df.columns if col.startswith('v')]
target_col = 'binary_label'
df_processed = df[feature_cols + [target_col]]


df_normal = df_processed[df_processed['binary_label'] == 0]
df_falling = df_processed[df_processed['binary_label'] == 1]

# 데이터셋 섞기
df_normal = df_normal.sample(frac=1).reset_index(drop=True)
df_falling = df_falling.sample(frac=1).reset_index(drop=True)

# 데이터셋 분할
val_ratio = 0.1
test_ratio = 0.2
train_ratio = 1 - (val_ratio + test_ratio)

data_len = len(df_normal)
normal_train_end = int(data_len * train_ratio)
normal_val_end = normal_train_end + int(data_len * val_ratio)

normal_train = df_normal[:normal_train_end]
normal_val = df_normal[normal_train_end:normal_val_end]
normal_test = df_normal[normal_val_end:]

data_len = len(df_falling)
falling_train_end = int(data_len * train_ratio)
falling_val_end = falling_train_end + int(data_len * val_ratio)

falling_train = df_falling[:falling_train_end]
falling_val = df_falling[falling_train_end:falling_val_end]
falling_test = df_falling[falling_val_end:]

train_df = pd.concat([normal_train, falling_train])
val_df = pd.concat([normal_val, falling_val])
test_df = pd.concat([normal_test, falling_test])

# 결과
print(f"훈련용 데이터셋(train): {len(train_df)}")
print(f"검증용 데이터셋(val): {len(val_df)}")
print(f"테스트용 데이터셋(test): {len(test_df)}")

# 저장
train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("validation_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)