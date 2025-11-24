import librosa
import numpy as np


# 센서 값 위치를 정하면 정해진 윈도우 크기의 앞 센서값을 불러오는 함수
def get_window(v, size, end):
    return v[end - size + 1 : end + 1]

# F0: V(t) - median (window: 50)
def calculate_Detrend(v):
    output = np.zeros(250)

    for t in range(250):
       t_in = t + 50
       window = get_window(v, 50, t_in)

       output[t] = v[t_in] - np.median(window)

    return output

# F1: MovingMAD (window: 50)
def calculate_MovingMAD(v):
    output = np.zeros(250)

    for t in range(250):
        t_in = t + 50
        window = get_window(v, 50, t_in)

        # MAD 계산
        abs_deviations = np.abs(window - np.median(window))
        mad = np.median(abs_deviations)

        output[t] = mad

    return output

# F2: MovingKurtosis (window: 50)
def calculate_MovingKurtosis(v):
    output = np.zeros(250)

    for t in range(250):
        t_in = t + 50
        window = get_window(v, 50, t_in)

        # Kurtosis 계산
        kurtosis = np.sum((((window - np.mean(window)) / np.std(window))**4) / len(window)) - 3

        output[t] = kurtosis

    return output

# F3: 미분값(기울기)
def calculate_Gradient(v):
    output = np.asarray(v, dtype=float)
    output = output[50:300] - output[49:299]

    return output

# F4: 창적분값 (window: 15)
def calculate_Integral(v):
    output = np.asarray(v, dtype=float)
    c = np.cumsum(output)
    sum = c[50:300] - c[35:285]

    return sum

# F5~F12: STFT (window: 50)
def calculate_STFT(v):
    n_fft = 50
    STFT_BANDS = [(1, 2), (2, 3), (3, 4), (4, 5),
                  (5, 6), (6, 7), (7, 8), (8, 15)]

    stft = librosa.stft(y=v.astype(float), n_fft=n_fft, win_length=50, hop_length=1, center=False)
    mag = np.abs(stft)
    mag = mag[:, 1:251]

    # 라이브러리에서 나온 각 밴드가 실제 출력 밴드 어디에 해당하는지 계산
    fft_freqs = librosa.fft_frequencies(sr=50, n_fft=n_fft)
    band_indices_list = []
    for f_min, f_max in STFT_BANDS:
        indices = np.where((fft_freqs >= f_min) & (fft_freqs < f_max))[0]
        band_indices_list.append(indices)

    output_map = np.zeros((250, len(STFT_BANDS)), dtype=np.float32)

    for i, band_indices in enumerate(band_indices_list):
        if len(band_indices) == 0:
            continue
        band_energy = mag[band_indices, :].sum(axis=0)
        output_map[:, i] = band_energy

    return output_map

# (300,) 의 데이터를 받고 (250, 13) 데이터로 바꾸는 처리 함수
def preprocess_data(v):
    feature_map = np.zeros((250, 13))

    feature_map[:, 0] = calculate_Detrend(v)
    feature_map[:, 1] = calculate_MovingMAD(v)
    feature_map[:, 2] = calculate_MovingKurtosis(v)
    feature_map[:, 3] = calculate_Gradient(v)
    feature_map[:, 4] = calculate_Integral(v)
    feature_map[:, 5:] = calculate_STFT(v)

    return feature_map

def create_feature(dataset):
    n_data = dataset.shape[0]

    X_processed = np.zeros((n_data, 250, 13))  # (N, 250, 13)의 빈 3D 텐서 생성
    for i in range(n_data):
        v = dataset[i, :]
        X_processed[i, :, :] = preprocess_data(v)

    return X_processed
