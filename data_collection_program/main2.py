import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import numpy as np
import time
from datetime import datetime
from threading import Thread
from collections import deque
import serial  # 시리얼 통신

# Matplotlib와 Tkinter 연동
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib import font_manager, rc

# --- [수정] 1. 시리얼 포트 설정 (사용자 수정 필요) ---
SERIAL_PORT = 'COM3'  # <-- 본인 환경에 맞게 수정
BAUD_RATE = 115200

# --- 윈도우 전용 한글 폰트 설정 ---

try:
    # 윈도우의 '맑은 고딕' 폰트를 기본 폰트로 설정
    rc('font', family='Malgun Gothic')
    print("'맑은 고딕' 폰트가 성공적으로 설정되었습니다.")
except Exception as e:
    # '맑은 고딕' 폰트를 찾지 못했을 경우
    print(f"폰트 설정 오류: {e}")
    print("'맑은 고딕' 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
    print("다른 폰트(예: 'NanumGothic', 'Dotum')로 시도해 보세요.")
    pass
# --- 폴더 이름 정의 ---
RAW_DATA_FOLDER = "raw_data"
NORMAL_FOLDER = "normal"
FALLING_FOLDER = "falling"
FAKE_FOLDER = "fake"  # <--- [수정] '넘어진 척' 폴더 추가

# --- 그래프 설정 ---
PLOT_WINDOW_SEC = 7.0
ANIMATION_INTERVAL_MS = 20


class DataLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PIR 낙상 데이터 수집기 (자동 연결)")
        self.root.geometry("1100x700")

        # 시리얼 및 데이터 변수
        self.ser = None
        self.ani = None
        self.last_valid_data = 2500.0

        self.current_file_path = None
        self.collection_window_size = 0
        self.sample_rate = 50  # 기본값 (Hz 입력창에서 읽어옴)
        self.plot_window_size = int(self.sample_rate * PLOT_WINDOW_SEC)

        self.is_collecting = False
        self.collection_buffer = []
        self.collection_step_counter = 0

        self.event_markers = []
        self.marker_lines = []
        
        # <--- [수정] 라벨별 카운터 변수 초기화
        self.normal_count = 0
        self.falling_count = 0
        self.fake_count = 0

        self.setup_folders()
        self.get_initial_counts() # <--- [수정] 프로그램 시작 시 기존 파일 수 로드

        # --- 2. GUI 레이아웃 생성 ---
        self.status_var = tk.StringVar(value="준비 중...")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill="x")

        left_frame = ttk.Frame(root, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        # --- 왼쪽 프레임 채우기 ---

        # [수정] 2-4. 수집 설정 프레임 (이제 첫 번째)
        settings_frame = ttk.LabelFrame(left_frame, text="1. 수집 설정")
        settings_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(settings_frame, text="샘플링 속도 (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.hz_var = tk.StringVar(value="50")
        self.entry_hz = ttk.Entry(settings_frame, textvariable=self.hz_var, width=10, state="disabled")  # 초기 비활성화
        self.entry_hz.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(settings_frame, text="수집 시간 (초):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sec_var = tk.StringVar(value="6")
        self.entry_sec = ttk.Entry(settings_frame, textvariable=self.sec_var, width=10, state="disabled")  # 초기 비활성화
        self.entry_sec.grid(row=1, column=1, padx=5, pady=5)

        self.lbl_sample_count = ttk.Label(settings_frame, text="= 총 250 샘플")
        self.lbl_sample_count.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.btn_start = ttk.Button(settings_frame, text="시작", command=self.start_collection,
                                      state="disabled")  # 초기 비활성화
        self.btn_start.grid(row=0, column=2, rowspan=3, padx=10, pady=5, ipady=15, sticky="e")

        # [수정] 2-5. 라벨링 프레임 (이제 두 번째)
        labeling_frame = ttk.LabelFrame(left_frame, text="2. 데이터 라벨링")
        labeling_frame.pack(fill="x")
        # (이하 동일)
        ttk.Label(labeling_frame, text="현재 파일:").pack(pady=5)
        self.lbl_current_file_var = tk.StringVar(value="(없음)")
        ttk.Label(labeling_frame, textvariable=self.lbl_current_file_var, foreground="blue",
                  font=("", 10, "bold")).pack(pady=(0, 10))
        button_frame = ttk.Frame(labeling_frame)
        button_frame.pack(pady=5, fill="x", expand=True)
        self.btn_discard = ttk.Button(button_frame, text="버리기", command=self.discard_file, state="disabled")
        self.btn_discard.pack(fill="x", expand=True, padx=5, pady=2)
        self.btn_normal = ttk.Button(button_frame, text="정상 (0)", command=self.label_normal, state="disabled")
        self.btn_normal.pack(fill="x", expand=True, padx=5, pady=2)
        self.btn_fall = ttk.Button(button_frame, text="넘어짐 (1)", command=self.label_fall, state="disabled")
        self.btn_fall.pack(fill="x", expand=True, padx=5, pady=2)
        
        # <--- [수정] '넘어진 척' 버튼 추가
        self.btn_fake = ttk.Button(button_frame, text="넘어진 척 (2)", command=self.label_fake, state="disabled")
        self.btn_fake.pack(fill="x", expand=True, padx=5, pady=2)


        # --- 오른쪽 프레임 (그래프) ---
        plot_frame = ttk.LabelFrame(right_frame, text="실시간 센서 (최근 7초)")
        plot_frame.pack(fill="both", expand=True, pady=(0, 5))
        self.fig1 = Figure(figsize=(7, 3.5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        static_plot_frame = ttk.LabelFrame(right_frame, text="수집된 데이터 (스냅샷)")
        static_plot_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.fig2 = Figure(figsize=(7, 3.5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=static_plot_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- 3. Matplotlib 초기화 (애니메이션 시작 X) ---
        self.init_plots()

        # --- 4. 이벤트 바인딩 ---
        self.hz_var.trace_add("write", self.update_sample_count)
        self.sec_var.trace_add("write", self.update_sample_count)
        self.update_sample_count()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- [수정] 5. 자동 시리얼 연결 시도 ---
        self.connect_serial()  # 프로그램 시작 시 바로 연결 시도

    def init_plots(self):
        """그래프 초기 설정"""
        self.update_sample_count()
        self.plot_buffer = deque([self.last_valid_data] * self.plot_window_size, maxlen=self.plot_window_size)
        self.plot_time_axis = np.linspace(0, PLOT_WINDOW_SEC, self.plot_window_size)

        self.ax1.clear()
        self.ax1.set_title("실시간 센서 데이터")
        self.ax1.set_xlabel("시간 (초)")
        self.ax1.set_ylabel("센서 값")
        self.ax1.set_ylim(0, 4096)
        self.ax1.set_xlim(0, PLOT_WINDOW_SEC)
        self.ax1.grid(True)
        self.line1, = self.ax1.plot(self.plot_time_axis, self.plot_buffer, lw=1.5)
        self.fig1.tight_layout(pad=0.5)
        self.canvas1.draw()

        self.ax2.clear()
        self.ax2.set_title("수집된 데이터 스냅샷")
        self.ax2.set_ylim(0, 4096)
        self.ax2.grid(True)
        self.fig2.tight_layout(pad=0.5)
        self.canvas2.draw()

    # --- [수정] 시리얼 연결 함수 ---
    def connect_serial(self):
        """(자동 호출) 시리얼 포트 연결 및 애니메이션 시작"""
        if self.ser and self.ser.is_open:
            # 이미 연결된 경우 (예: 재시작 시) 무시
            return

        self.status_var.set(f"'{SERIAL_PORT}'에 연결 시도 중...")
        try:
            # 설정된 Hz 값을 읽어옴
            self.sample_rate = int(self.hz_var.get())
            if self.sample_rate <= 0: raise ValueError("Hz > 0")

            # 상수 값으로 연결 시도
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.ser.flushInput()

            self.status_var.set(f"'{SERIAL_PORT}'에 연결됨 ({self.sample_rate}Hz). 그래프 시작.")

            # 연결 성공 시 GUI 활성화
            self.set_gui_state(is_collecting=False)  # 연결 성공 + 수집 아님 상태
            self.entry_hz.config(state="normal")
            self.entry_sec.config(state="normal")
            self.btn_start.config(state="normal")

            # 그래프 재초기화 및 애니메이션 시작
            self.init_plots()
            self.ani = FuncAnimation(self.fig1, self.update_plot,
                                       interval=ANIMATION_INTERVAL_MS, blit=False, cache_frame_data=False)

        except ValueError as e:
            messagebox.showerror("입력 오류", f"샘플링 속도(Hz)는 양의 숫자여야 합니다.\n{e}")
            self.status_var.set("오류: Hz 값을 확인하세요.")
            self.ser = None
        except serial.SerialException as e:
            messagebox.showerror("연결 오류", f"포트 '{SERIAL_PORT}' 연결에 실패했습니다:\n{e}")
            self.status_var.set("오류: 시리얼 포트를 확인하세요.")
            self.ser = None
        except Exception as e:
            messagebox.showerror("오류", f"예상치 못한 오류:\n{e}")
            self.status_var.set("오류 발생.")
            self.ser = None

    # --- [수정] 연결 해제 함수 ---
    def disconnect_serial(self):
        """시리얼 포트 연결 해제 및 애니메이션 중지"""
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None

        if self.ser and self.ser.is_open:
            self.ser.close()

        self.ser = None
        self.status_var.set("연결 해제됨.")
        # GUI 비활성화
        self.set_gui_state(is_collecting=False)  # 연결 끊김 + 수집 아님 상태

    def on_closing(self):
        """창을 닫을 때 시리얼 포트 정리"""
        self.disconnect_serial()
        self.root.destroy()

    def setup_folders(self):
        # <--- [수정] FAKE_FOLDER 추가
        for folder in [RAW_DATA_FOLDER, NORMAL_FOLDER, FALLING_FOLDER, FAKE_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    # <--- [수정] 기존 파일 카운트 로드 함수
    def get_initial_counts(self):
        """프로그램 시작 시 각 라벨 폴더의 파일 수를 셉니다."""
        try:
            self.normal_count = len(os.listdir(NORMAL_FOLDER))
            self.falling_count = len(os.listdir(FALLING_FOLDER))
            self.fake_count = len(os.listdir(FAKE_FOLDER))
            print(f"초기 카운트: Normal={self.normal_count}, Falling={self.falling_count}, Fake={self.fake_count}")
        except Exception as e:
            print(f"초기 카운트 로드 오류: {e}")
            # 오류 발생 시 0으로 강제 초기화
            self.normal_count = 0
            self.falling_count = 0
            self.fake_count = 0


    def update_sample_count(self, *args):
        """Hz/Sec 변경 시 윈도우 크기를 업데이트합니다."""
        # (이전 코드와 동일)
        try:
            sample_rate = int(self.hz_var.get())
            collection_sec = float(self.sec_var.get())
            self.collection_window_size = int(sample_rate * collection_sec)
            self.lbl_sample_count.config(text=f"= 총 {self.collection_window_size} 샘플")
            plot_window_size = int(sample_rate * PLOT_WINDOW_SEC)
            if not self.is_collecting and self.plot_window_size != plot_window_size:
                self.plot_window_size = plot_window_size
                self.plot_buffer = deque([self.last_valid_data] * self.plot_window_size, maxlen=self.plot_window_size)
                self.plot_time_axis = np.linspace(0, PLOT_WINDOW_SEC, self.plot_window_size)
        except ValueError:
            self.lbl_sample_count.config(text="= 계산 오류")

    def get_sensor_data(self):
        """실제 시리얼 포트에서 데이터를 읽어옵니다."""
        # (이전 코드와 동일)
        if not (self.ser and self.ser.is_open):
            return self.last_valid_data
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    value = int(line)
                    self.last_valid_data = value
        except (ValueError, UnicodeDecodeError) as e:
            print(f"시리얼 데이터 파싱 오류: {e}")
        except serial.SerialException as e:
            messagebox.showerror("시리얼 오류", f"장치 연결이 끊어졌습니다:\n{e}")
            self.disconnect_serial()
        return self.last_valid_data

    def update_plot(self, frame):
        """Matplotlib 애니메이션 콜백 함수"""
        # (이전 코드와 동일)
        new_data = self.get_sensor_data()
        self.plot_buffer.append(new_data)
        for line in self.marker_lines: line.remove()
        self.marker_lines.clear()
        self.event_markers = [idx - 1 for idx in self.event_markers if idx > 0]
        for idx in self.event_markers:
            if idx < len(self.plot_time_axis):
                plot_x = self.plot_time_axis[idx]
                line = self.ax1.axvline(x=plot_x, color='red', linestyle='--')
                self.marker_lines.append(line)
        self.line1.set_ydata(self.plot_buffer)
        if self.is_collecting:
            self.collection_buffer.append(new_data)
            self.collection_step_counter += 1
            if self.collection_step_counter >= self.collection_window_size:
                self.stop_collection()
        return self.line1,

    def start_collection(self):
        """데이터 수집 시작"""
        # (이전 코드와 동일)
        if self.is_collecting: return
        try:
            self.sample_rate = int(self.hz_var.get())
            collection_sec = float(self.sec_var.get())
            self.collection_window_size = int(self.sample_rate * collection_sec)
            self.plot_window_size = int(self.sample_rate * PLOT_WINDOW_SEC)
        except ValueError:
            messagebox.showerror("오류", "Hz와 초는 숫자여야 합니다.")
            return
        if self.collection_window_size <= 0:
            messagebox.showerror("오류", "샘플 수가 0보다 커야 합니다.")
            return

        self.root.bell()  # <<< [수정] 수집 시작 시 경고음

        self.set_gui_state(is_collecting=True)
        self.status_var.set(f"수집 중... ({self.collection_window_size} 샘플)")
        self.is_collecting = True
        self.collection_buffer.clear()
        self.collection_step_counter = 0
        self.event_markers.append(self.plot_window_size - 1)
        self.ax2.clear()
        self.ax2.set_title("수집 중...")
        self.ax2.set_ylim(0, 4096)
        self.ax2.grid(True)
        self.canvas2.draw()

    def stop_collection(self):
        """데이터 수집 종료"""
        self.root.bell()  # <<< [수정] 수집 종료 시 경고음

        # (이전 코드와 동일)
        self.is_collecting = False
        self.event_markers.append(self.plot_window_size - 1)
        collected_data = list(self.collection_buffer)
        self.draw_static_plot(collected_data)
        self.save_data_to_file(collected_data)
        self.set_gui_state(is_collecting=False)

    def draw_static_plot(self, data):
        """하단(ax2) 그래프에 수집된 데이터를 그립니다."""
        # (이전 코드와 동일)
        self.ax2.clear()
        self.ax2.set_title(f"수집 완료 ({len(data)} 샘플)")
        self.ax2.set_ylim(0, 4096)
        self.ax2.grid(True)
        self.ax2.plot(data, color='blue')
        self.fig2.tight_layout(pad=0.5)
        self.canvas2.draw()

    def save_data_to_file(self, data):
        """수집된 데이터를 raw_data 폴더에 저장합니다."""
        # (이전 코드와 동일)
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}.csv"
            filepath = os.path.join(RAW_DATA_FOLDER, filename)
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
            self.current_file_path = filepath
            self.lbl_current_file_var.set(filename)
            self.status_var.set("수집 완료! 라벨링 대기 중.")
        except Exception as e:
            messagebox.showerror("파일 저장 오류", str(e))
            self.status_var.set("파일 저장 중 오류 발생.")

    # --- [수정] GUI 상태 관리 함수 ---
    def set_gui_state(self, is_collecting):
        """GUI 위젯의 활성화/비활성화 상태를 변경합니다."""

        # 시리얼 연결 상태 (self.ser가 None이 아니면 연결된 것으로 간주)
        is_connected = self.ser is not None and self.ser.is_open

        # 수집 버튼/입력창 상태
        start_state = "normal" if is_connected and not is_collecting else "disabled"
        entry_state = "normal" if is_connected and not is_collecting else "disabled"

        self.btn_start.config(state=start_state)
        self.entry_hz.config(state=entry_state)
        self.entry_sec.config(state=entry_state)

        # 라벨 버튼 상태
        label_state = "normal" if is_connected and not is_collecting and self.current_file_path else "disabled"
        self.btn_discard.config(state=label_state)
        self.btn_normal.config(state=label_state)
        self.btn_fall.config(state=label_state)
        self.btn_fake.config(state=label_state) # <--- [수정] '넘어진 척' 버튼 상태 관리

    # --- 라벨링 함수 (한글 메시지) ---
    def process_file(self, label):
        """파일을 라벨링(이동/삭제)하는 공통 로직입니다."""
        # (이전 코드와 동일)
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            messagebox.showwarning("오류", "처리할 파일이 없습니다.")
            return
        original_filename = os.path.basename(self.current_file_path)
        try:
            if label == -1:
                os.remove(self.current_file_path)
                status_message = f"파일 '{original_filename}' 삭제됨."
            else:
                # --- [수정] 분류, 카운팅, 이름 변경 로직 ---
                
                # 1. 라벨에 따라 폴더, 접두사, 카운터 결정
                if label == 0:
                    self.normal_count += 1
                    count = self.normal_count
                    dest_folder = NORMAL_FOLDER
                    prefix = "normal"
                elif label == 1:
                    self.falling_count += 1
                    count = self.falling_count
                    dest_folder = FALLING_FOLDER
                    prefix = "falling"
                elif label == 2: # <--- [수정] '넘어진 척' 라벨(2) 처리
                    self.fake_count += 1
                    count = self.fake_count
                    dest_folder = FAKE_FOLDER
                    prefix = "fake"
                else:
                    return # 알 수 없는 라벨
                
                # 2. 새 파일명 생성 (예: "falling_5번째_20251031_160930.csv")
                new_filename = f"{prefix}_{count}번째_{original_filename}"
                
                # 3. 최종 저장 경로 설정
                dest_path = os.path.join(dest_folder, new_filename)
                
                # (원본 파일 읽기)
                with open(self.current_file_path, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    data_row = next(reader)
                    data_row.append(label) # 데이터 끝에 라벨 추가
                
                # (새 파일명으로 대상 폴더에 쓰기)
                with open(dest_path, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(data_row)
                
                # 5. 원본 파일 삭제
                os.remove(self.current_file_path)
                
                # 6. 상태 메시지 업데이트
                status_message = f"-> '{new_filename}'(으)로 저장됨."
            
            # (공통 GUI 업데이트 로직)
            self.current_file_path = None
            self.lbl_current_file_var.set("(없음)")
            self.status_var.set(status_message)
            self.set_gui_state(is_collecting=False)
        except Exception as e:
            messagebox.showerror("파일 처리 오류", str(e))
            self.status_var.set("파일 처리 중 오류 발생.")

    def discard_file(self):
        self.process_file(label=-1)

    def label_normal(self):
        self.process_file(label=0)

    def label_fall(self):
        self.process_file(label=1)
        
    # <--- [수정] '넘어진 척' 콜백 함수
    def label_fake(self):
        self.process_file(label=2)


if __name__ == "__main__":
    root = tk.Tk()
    app = DataLabelingApp(root)
    root.mainloop()