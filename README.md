# 🏃‍♂️ Real-time Fall Detection System (25-2 ML Team)

![Project Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![ESP32](https://img.shields.io/badge/Hardware-ESP32-red)

## **💡 Project Overview** 
> **이 프로젝트는 ESP32 센서와 머신러닝을 활용하여 실시간으로 낙상(Fall)을 감지하고 분류하는 시스템입니다.**
> **직접 수집한 데이터를 기반으로 데이터 증강 및 전처리를 거쳐 모델을 학습시켰으며, 실시간 시스템까지 구축하였습니다.**

> **수업 - 기계학습기초**

---

## 📸 Performance & Demo
<div align="center">
  <img src="https://github.com/user-attachments/assets/69c9d4ef-566c-4ede-803f-81d906e569a9" alt="Performance Graph" width="45%">
  <img src="https://github.com/user-attachments/assets/794bff96-5aca-4f07-993c-bbba2a22853f" alt="Hardware Setup" width="45%">
</div>

---

## 📂 Project Structure

### 1️⃣ Data Collection Program
**데이터 수집을 위한 임베디드 소프트웨어 및 수집기입니다.**

> ⚠️ **Hardware Requirement** > 이 폴더의 코드는 **ESP32 보드와 센서**가 연결된 환경에서만 정상 작동합니다.

- **`collect.py`** 실시간 데이터 수집 프로그램입니다. 
- **`esp32_software.ino`** ESP32 보드용 펌웨어입니다. 센서 데이터를 `50Hz`(초당 50회)로 샘플링하여 시리얼로 전송합니다.
- **Raw Data Folders** - 직접 수집한 Raw 데이터가 저장되어 있습니다.
  - 📁 `fake`: 낙상과 유사하지만 낙상이 아닌 행동
  - 📁 `falling`: 실제 낙상 데이터
  - 📁 `normal`: 일상적인 행동 데이터

<br>

### 2️⃣ Data Process Program
**데이터 전처리, 병합, 증강을 담당하는 파이프라인입니다.**

- **`data_merge.py`**: 분산된 데이터를 하나로 합쳐 통합 데이터셋(`all_dataset.csv`)을 생성합니다.
- **`all_dataset.csv`**: 전처리가 완료된 통합 데이터셋 파일입니다.
- **`augment_functions.py`**: 데이터 불균형 해소를 위한 증강 함수 모음입니다.
- **`preprocess_functions.py`**: 노이즈 제거 및 정규화 등 전처리 함수 모음입니다.
- **`load_dataset.py`**: 파일을 읽어 데이터셋을 로드하는 함수 모음입니다.
  - `load_dataset()`: 전체 데이터 로드  
  - `split_dataset()`: Train / Validation / Test 데이터 분할

<br>

### 3️⃣ Model Train
**딥러닝 모델 학습 및 테스트를 위한 공간입니다.**

| 파일명 | 설명 | 비고 |
|:---:|:---|:---|
| **train_test_v1.ipynb** | 첫 번째 훈련 - 모델 구조 및 하이퍼파라미터 튜닝용 파일 | Colab 권장 |
| **train_test_v2.ipynb** | 두 번째 훈련 - K-Fold 교차 검증 테스트용 파일 | Colab 권장 |
| **train.py** | **최종 학습 코드** (로컬 실행) | `best_fall_model.keras`, `robust_scaler.pkl` 생성 |
- 주피터 노트북 파일들의 경우 all_dataset.csv이 있어야 합니다.
- 실제 학습은 로컬에서 진행하였습니다.
> **📝 Note** > `train.py` 실행 시 39번째 행의 데이터 경로(`all_data.csv`)를 본인의 환경에 맞게 수정해야 합니다.


<br>

### 4️⃣ Realtime Test Program
**최종 산출물 및 실시간 테스트 프로그램입니다.**

- **`best_fall_model.keras`**: 학습이 완료된 최종 모델 파일
- **`robust_scaler.pkl`**: 학습 데이터에 맞춰진 스케일러 파일
- **`realtime.py`**: 실시간 낙상 감지 테스트 프로그램 
  - *Note: 실제 센서 하드웨어 연결 필요*

---

## 🛠 Tech Stack
- **Hardware**: ESP32
- **Software**: Python, Arduino IDE
- **Libraries**: TensorFlow, Pandas, NumPy, Scikit-learn, PySerial

---

## 👨‍💻 Contributors
- **25-2 ML Team**
- [![Role](https://img.shields.io/badge/Leader-suhyeonx-green)](https://github.com/suhyeonx)
- [![Role](https://img.shields.io/badge/Member-yeevdev-green)](https://github.com/yeevdev)
- [![Role](https://img.shields.io/badge/Member-minjoooo0-green)](https://github.com/minjoooo0)
- [![Role](https://img.shields.io/badge/Member-sbddjt-green)](https://github.com/sbddjt)
