import numpy as np
import random as pyrandom


# 스파이크 노이즈 추가 함수
def make_spike_noise(v_data):
    v_new = np.array(v_data, dtype=int)
    data_len = len(v_new)

    # 무작위 위치 선택
    duration = pyrandom.randint(1, 3)
    start_idx = pyrandom.randint(0, data_len - duration)
    end_idx = start_idx + duration

    if pyrandom.random() < 0.5:  # 하한 스파이크
        value = pyrandom.randint(1, 10)
    else:  # 상한 스파이크
        value = pyrandom.randint(2900,3000)


    v_new[start_idx:end_idx] = value

    return v_new

# 가우시안 노이즈 추가 함수
def make_gaussian_noise(v_data):
    v_new = np.array(v_data, dtype=float)

    # 가우시안 노이즈 생성
    noise = np.random.normal(scale=10.0, size=v_new.shape)
    v_new = v_new + noise

    # 데이터 보정(최대/최소, 자료형)
    v_new = np.clip(v_new, 1.0, 3000.0)
    v_new = v_new.astype(int)

    return v_new

# 시간 왜곡 함수
def make_time_warping(v_data):
    v_new = np.array(v_data, dtype=float)
    data_len = len(v_new)

    # 배속 설정
    scale = pyrandom.uniform(0.8, 1.2)
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

# 시간축 이동 증강 함수
def make_time_shift(v_data):
    v_new = np.array(v_data, dtype=int)

    shift = pyrandom.randint(-10, 10)

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

# 스케일 변화 증강 함수
def make_scale_different(v_data):
    v_new = np.array(v_data, dtype=float)

    mean = np.mean(v_data)
    v_new = v_new - mean

    # 새로운 랜덤 거리 모사
    distance_new = pyrandom.uniform(1, 2.5)
    scale_factor = (1.5 / distance_new)**2
    v_new = v_new * scale_factor

    v_new = v_new + mean
    v_new = np.clip(v_new, 1.0, pyrandom.uniform(2950,3000))
    v_new = v_new.astype(int)

    return v_new

# 특정 데이터(v0~v299)에 대해서 랜덤한 증강(1개~3개)을 적용하는 함수
def augment(X, factor, label):
    n_to_add = len(X) * (factor - 1)
    X_augmented = []

    aug_functions = [make_spike_noise, make_gaussian_noise, make_time_warping, make_time_shift, make_scale_different]

    for _ in range(n_to_add):
        sample_data = X[pyrandom.randint(0, len(X) - 1)].copy()

        n_to_augment = pyrandom.randint(1, 3)
        selected_functions = pyrandom.sample(aug_functions, n_to_augment)

        for func in selected_functions:
            sample_data = func(sample_data)

        X_augmented.append(sample_data)

    X_augmented = np.array(X_augmented)
    y_augmented = np.full(len(X_augmented), label, dtype=int)

    return X_augmented, y_augmented

# 데이터 셋에 대해서 정한 배율만큼 데이터를 증강시키는 함수
def augment_dataset(X, y, normal_aug, falling_aug):
    # 클래스별로 데이터 분리
    X_normal = X[y == 0]
    X_falling = X[y == 1]

    X_augmented = [X]
    y_augmented = [y]

    # 증강
    X_normal_augmented, y_normal_augmented = augment(X_normal, normal_aug, 0)
    X_augmented.append(X_normal_augmented)
    y_augmented.append(y_normal_augmented)

    X_falling_augmented, y_falling_augmented = augment(X_falling, falling_aug, 1)
    X_augmented.append(X_falling_augmented)
    y_augmented.append(y_falling_augmented)

    # 증강된 데이터셋 정리 및 셔플
    X_final = np.vstack(X_augmented)
    y_final = np.concatenate(y_augmented)

    rand_idx = np.random.permutation(len(X_final))
    X_final = X_final[rand_idx]
    y_final = y_final[rand_idx]

    return X_final, y_final