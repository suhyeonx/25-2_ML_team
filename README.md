# 25-2_ML_team

- `data_collection_program`: 데이터 수집 관련 프로그램과 수집한 실제 raw 데이터가 들어있는 폴더입니다.
    - `collect.py`: 실시간 데이터 수집프로그램입니다.
        - AI의 도움을 받았습니다.
        - 실행 시 실제 하드웨어(ESP32보드 및 센서)가 필요하므로 실행이 되지 않을 수 있습니다.
    - `esp32_software.ino`: 실제 하드웨어(ESP32보드)에 들어가는 펌웨어입니다.
        - 센서 데이터를 시리얼 형식으로 받아 50Hz 샘플링(초당 50번)으로 전송합니다.
    - `fake`, `falling`, `normal`: 실제 저희가 직접 수집한 raw 데이터들이 모여있습니다.
    
- `data_process_program`: 데이터 처리 관련 프로그램과 전체 데이터셋이 들어있는 폴더입니다.
    - `data_merge.py`: 모든 데이터를 합쳐 통합 데이터셋 파일을 만듭니다.
    - `all_dataset.csv`: 위의 결과로 만들어진 통합 데이터셋 파일입니다.
    - `load_dataset.py`: `all_dataset.csv` 파일을 읽어 데이터셋을 로드하는 프로그램
        - `load_dataset()`: 전체 데이터셋을 로드합니다.
        - `split_dataset()`: 데이터셋을 train, val, test로 쪼갭니다.
    - `augment_functions.py`: 데이터셋 증강 프로그램
        - 데이터셋을 증강하는 함수들이 작성되어 있습니다.
    - `preprocess_functions.py`:
        - 데이터를 전처리하는 함수들이 작성되어 있습니다.

- `model_train`: 모델 및 학습관련 파일들이 들어있는 폴더입니다.
    - `train_test_v1/train_test_v1.ipynb`: 첫번째 모델들 테스트(모델 구조 및 하이퍼파라미터)용 주피터 노트북 파일입니다.
        - Google Colab에서 `all_dataset.csv` 파일이 있어야 작동합니다.
        - 또는 전체 프로젝트 폴더가 그대로 있다는 가정 하에 `train_test_v1.py` 파일을 실행해도 됩니다.
    - `train_test_v2/train_test_v2.ipynb`: 두번째 K-Fold 테스트용 주피터 노트북 파일입니다.
        - Google Colab에서 `all_dataset.csv` 파일이 있어야 작동합니다.
        - 또는 전체 프로젝트 폴더가 그대로 있다는 가정 하에 `train_test_v2.py` 파일을 실행해도 됩니다.
    - `train.py`: 테스트로 얻은 최적의 모델을 실제 학습하는 코드입니다.
        - 코드 실행시 `best_fall_model.keras`, `robust_scaler.pkl` 파일이 생성됩니다.
        - `39`번째 행의 경로를 알맞은 `all_data.csv`파일로 설정하면 학습이 실행됩니다.
        - 실제 학습은 로컬에서 진행하였습니다.

- `realtime_test_program`: 실시간 테스트용 프로그램과 최종 모델이 들어있는 폴더입니다.
    - `best_fall_model.keras`: 최종 모델 파일입니다.
    - `robust_scaler.pkl`: 최종 스케일러 파일입니다.
    - `realtime.py`: 실시간 테스트 프로그램입니다.
        - AI의 도움을 받았습니다. 
        - 실행 시 실제 하드웨어(ESP32보드 및 센서)가 필요하므로 실행이 되지 않을 수 있습니다.
