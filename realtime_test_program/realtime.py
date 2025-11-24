# 이 코드는 AI의 도움을 받았습니다.

import tkinter as tk
from tkinter import font
import threading
import librosa
import numpy as np
import tensorflow as tf
import joblib
import serial
import time
import collections
import queue
from data_process_program.preprocess_functions import *

# =========================================================
# [설정] 환경 변수 (본인 환경에 맞게 수정 필수)
# =========================================================
SERIAL_PORT = '/dev/cu.usbserial-0001'  # 맥/리눅스 예시 ('COM3' 등 윈도우 포트 확인)
BAUD_RATE = 115200
MODEL_PATH = 'best_fall_model.keras'
SCALER_PATH = 'robust_scaler.pkl'

THRESHOLD = 0.5
ALARM_HOLD_TIME = 5.0  # 낙상 감지 시 10초간 경고 화면 유지


# =========================================================
# [시스템 클래스] GUI + 로직 통합
# =========================================================
class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 실시간 낙상 감지 시스템")
        self.root.geometry("800x600")
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        # 1. 모델 로드
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"오류: {e}")
            self.root.destroy()
            return

        # 2. 변수 초기화
        self.alarm_end_time = 0
        self.current_prob = 0.0
        self.last_sensor_val = 0.0  # 실시간 센서값 저장용
        self.running = True

        # 3. 시작
        self.setup_ui()
        self.serial_thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.serial_thread.start()
        self.update_ui_loop()

    def setup_ui(self):
        self.font_large = font.Font(family="Helvetica", size=80, weight="bold")
        self.font_small = font.Font(family="Helvetica", size=30, weight="bold")

        self.main_frame = tk.Frame(self.root, bg="#2ecc71")
        self.main_frame.pack(fill="both", expand=True)

        # 상태 메시지 (정상/낙상)
        self.status_label = tk.Label(self.main_frame, text="정상",
                                     font=self.font_large, bg="#2ecc71", fg="white")
        self.status_label.place(relx=0.5, rely=0.4, anchor="center")

        # 정보 메시지 (확률 + 센서값)
        self.info_label = tk.Label(self.main_frame, text="초기화 중...",
                                   font=self.font_small, bg="#2ecc71", fg="white")
        self.info_label.place(relx=0.5, rely=0.7, anchor="center")

    def update_ui_loop(self):
        """ 화면 갱신 로직 """
        current_time = time.time()

        # 1. 배경색 결정 (알람 시간 남았으면 빨강, 아니면 초록)
        if current_time < self.alarm_end_time:
            bg_color = "#e74c3c"  # 빨강
            status_text = "🚨 낙상 감지! 🚨"
        else:
            bg_color = "#2ecc71"  # 초록
            status_text = "정상 (Safe)"

        # 2. 텍스트 구성 (확률 + 센서값 같이 표시)
        # 예: "확률: 12.5% | 센서값: 512"
        info_text = f"확률: {self.current_prob * 100:.1f}%  |  센서값: {int(self.last_sensor_val)}"

        # 3. 적용
        self.main_frame.config(bg=bg_color)
        self.status_label.config(text=status_text, bg=bg_color)
        self.info_label.config(text=info_text, bg=bg_color)

        self.root.after(100, self.update_ui_loop)

    def serial_loop(self):
        window_size = 300
        buffer = collections.deque(maxlen=window_size)
        predict_cnt = 0

        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            ser.flushInput()
        except:
            return

        while self.running:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue

                    # 센서값 파싱
                    val = float(line.split(',')[0])

                    # ★ 화면 표시용 변수에 즉시 저장 (여기서 저장해야 GUI에 바로 뜸)
                    self.last_sensor_val = val

                    buffer.append(val)

                    if len(buffer) == window_size:
                        predict_cnt += 1
                        if predict_cnt >= 10:
                            predict_cnt = 0

                            # 예측 수행
                            raw_data = np.array(buffer)
                            features = preprocess_data(raw_data)
                            f_scaled = self.scaler.transform(features.reshape(-1, 13)).reshape(250, 13)

                            prob = self.model.predict(np.expand_dims(f_scaled, axis=0), verbose=0)[0][0]

                            self.current_prob = prob

                            # 알람 트리거
                            if prob > THRESHOLD:
                                self.alarm_end_time = time.time() + ALARM_HOLD_TIME

                except:
                    continue
# =========================================================
# [메인 실행]
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FallDetectionApp(root)


    # 창 닫을 때 스레드 종료 처리
    def on_closing():
        app.running = False
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()