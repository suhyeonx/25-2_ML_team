import numpy as np
import random
import pandas as pd


def make_spike_noise(v_data):
    v_new = np.array(v_data, dtype=int)
    data_len = len(v_new)

    # 무작위 위치 선택
    duration = random.randint(1, 3)
    start_idx = random.randint(0, data_len - duration)
    end_idx = start_idx + duration

    if random.random() < 0.5:  # 하한 스파이크
        value = random.randint(1, 10)
    else:  # 상한 스파이크
        value = random.randint(2900,3000)


    v_new[start_idx:end_idx] = value

    return v_new

def make_gaussian_noise(v_data):
    v_new = np.array(v_data, dtype=float)

    # 가우시안 노이즈 생성
    noise = np.random.normal(scale=10.0, size=v_new.shape)
    v_new = v_new + noise

    # 데이터 보정(최대/최소, 자료형)
    v_new = np.clip(v_new, 1.0, 3000.0)
    v_new = v_new.astype(int)

    return v_new

def make_time_warping(v_data):
    v_new = np.array(v_data, dtype=float)
    data_len = len(v_new)

    # 배속 설정
    scale = random.uniform(0.8, 1.2)
    new_len = int(data_len * scale)

    x_axis = np.arange(data_len)
    x_axis_new = np.linspace(0, data_len - 1, new_len)
    v_scaled = np.interp(x_axis_new, x_axis, v_new)

    v_final = np.zeros(data_len, dtype=float)

    if new_len > data_len:
        cut = new_len - data_len
        start_idx = cut // 2
        end_idx = start_idx + data_len

        v_final = v_scaled[start_idx:end_idx]
    else:
        padding = data_len - new_len
        padding_left = padding // 2
        padding_right = padding - padding_left

        v_final[padding_left:padding_left+new_len] = v_scaled
        v_final[:padding_left] = v_scaled[0]
        v_final[padding_left+new_len:] = v_scaled[-1]

    v_final = v_final.astype(int)

    return v_final

def make_time_shift(v_data):
    v_new = np.array(v_data, dtype=int)

    shift = random.randint(-10, 10)

    if shift == 0:
        return v_new

    v_final = np.zeros_like(v_new, dtype=int)

    if shift > 0:
        v_final[shift:] = v_new[:-shift]
        v_final[:shift] = v_new[0]
    else:
        v_final[:shift] = v_new[-shift:]
        v_final[shift:] = v_new[-1]

    return v_final

dataset_file = "../split/train_dataset.csv"
output_file = "augmented_data/train_augmented.csv"

normal_aug = 4
falling_aug = 8

aug_functions = [make_spike_noise, make_gaussian_noise, make_time_warping, make_time_shift]

df = pd.read_csv(dataset_file)
feature_cols = [f'v{i}' for i in range(300)]
label_col = 'binary_label'

df_normal = df[df[label_col] == 0]
df_falling = df[df[label_col] == 1]
print(f"normal 데이터 개수 : {len(df_normal)}개")
print(f"falling 데이터 개수 : {len(df_falling)}개")

new_data_list = []

# normal 데이터 증강
num_to_add_normal = len(df_normal) * (normal_aug - 1)
normal_np = df_normal[feature_cols].to_numpy()

for _ in range(num_to_add_normal):
    sample_row = normal_np[random.randint(0, len(normal_np) - 1)]
    aug_func = random.choice(aug_functions)

    new_row = aug_func(sample_row)

    new_row_dict = {col: val for col, val in zip(feature_cols, new_row)}
    new_row_dict[label_col] = 0
    new_data_list.append(new_row_dict)

# falling 데이터 증강
num_to_add_falling = len(df_falling) * (falling_aug - 1)
falling_np = df_falling[feature_cols].to_numpy()

for _ in range(num_to_add_falling):
    sample_row = falling_np[random.randint(0, len(falling_np) - 1)]
    aug_func = random.choice(aug_functions)

    new_row = aug_func(sample_row)

    new_row_dict = {col: val for col, val in zip(feature_cols, new_row)}
    new_row_dict[label_col] =  1
    new_data_list.append(new_row_dict)

df_augmented = pd.DataFrame(new_data_list)
df_final = pd.concat([df, df_augmented])
df_final = df_final.sample(frac=1).reset_index(drop=True)
df_final.to_csv(output_file, index=False, header=True)
print(f"최종 데이터 개수: {df_final[label_col].value_counts()}")