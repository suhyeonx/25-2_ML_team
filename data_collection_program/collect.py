# 이 코드는 AI의 도움으로 생성되었습니다. (This code was generated with the help of AI.)

import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import numpy as np
from datetime import datetime
from collections import deque
import serial

# Matplotlib + Tkinter 연동
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# ==============================
# 기본 설정 상수
# ==============================

# 시리얼 포트 설정 (환경에 맞게 수정)
SERIAL_PORT = '/dev/cu.usbserial-0001'
BAUD_RATE = 115200

# 데이터 저장용 폴더 이름
RAW_DATA_FOLDER = "raw_data"
NORMAL_FOLDER = "normal"
FALLING_FOLDER = "falling"
FAKE_FOLDER = "fake"

# 실시간 그래프에 보여줄 시간(초)
PLOT_WINDOW_SEC = 7.0

# 그래프 갱신 주기 (밀리초)
ANIMATION_INTERVAL_MS = 20


class DataLabelingApp:
    # ===========================================
    # 생성자: GUI 구성 & 초기 상태 설정
    # ===========================================
    def __init__(self, root):
        # --- 메인 윈도우 설정 ---
        self.root = root
        self.root.title("PIR 낙상 데이터 수집기 (자동 연결)")
        self.root.geometry("1100x700")

        # --- 시리얼 및 상태 관련 변수 ---
        self.ser = None                # 시리얼 포트 객체
        self.ani = None                # Matplotlib 애니메이션 객체
        self.last_valid_data = 2500.0  # 센서 값이 없을 때 기본값

        # --- 수집 관련 변수 ---
        self.current_file_path = None          # 방금 수집한 RAW csv 파일 경로
        self.collection_window_size = 0        # 한 번 수집할 샘플 개수
        self.sample_rate = 50                  # 샘플링 속도 (Hz)
        self.plot_window_size = int(self.sample_rate * PLOT_WINDOW_SEC)
        self.is_collecting = False             # 현재 수집 중인지 여부
        self.collection_buffer = []            # 한 번 수집되는 샘플 임시 저장
        self.collection_step_counter = 0       # 현재까지 수집된 샘플 수

        # --- 라벨별 파일 개수 (파일 이름에 번호 붙일 때 사용) ---
        self.normal_count = 0
        self.falling_count = 0
        self.fake_count = 0

        # --- 폴더 생성 및 기존 파일 개수 세기 ---
        for folder in [RAW_DATA_FOLDER, NORMAL_FOLDER, FALLING_FOLDER, FAKE_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        try:
            self.normal_count = len(os.listdir(NORMAL_FOLDER))
            self.falling_count = len(os.listdir(FALLING_FOLDER))
            self.fake_count = len(os.listdir(FAKE_FOLDER))
        except Exception:
            self.normal_count = 0
            self.falling_count = 0
            self.fake_count = 0

        # --- 상태 표시 라벨 (창 맨 아래) ---
        self.status_var = tk.StringVar(value="준비 중...")
        ttk.Label(root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill="x")

        # --- 메인 레이아웃: 왼쪽(설정/라벨링), 오른쪽(그래프) ---
        left_frame = ttk.Frame(root, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                         padx=(0, 10), pady=10)

        # ===========================================
        # 1. 왼쪽 영역: 수집 설정
        # ===========================================
        settings_frame = ttk.LabelFrame(left_frame, text="1. 수집 설정")
        settings_frame.pack(fill="x", pady=(0, 10))

        # --- 샘플링 속도 입력 (Hz) ---
        ttk.Label(settings_frame, text="샘플링 속도 (Hz):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        self.hz_var = tk.StringVar(value="50")
        self.entry_hz = ttk.Entry(settings_frame, textvariable=self.hz_var,
                                  width=10, state="disabled")
        self.entry_hz.grid(row=0, column=1, padx=5, pady=5)

        # --- 수집 시간 입력 (초) ---
        ttk.Label(settings_frame, text="수집 시간 (초):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        self.sec_var = tk.StringVar(value="6")
        self.entry_sec = ttk.Entry(settings_frame, textvariable=self.sec_var,
                                   width=10, state="disabled")
        self.entry_sec.grid(row=1, column=1, padx=5, pady=5)

        # --- 총 샘플 수 표시 ---
        self.lbl_sample_count = ttk.Label(settings_frame, text="= 총 300 샘플")
        self.lbl_sample_count.grid(row=2, column=0, columnspan=2,
                                   padx=5, pady=5, sticky="w")

        # --- 수집 시작 버튼 ---
        self.btn_start = ttk.Button(settings_frame, text="시작",
                                    command=self.start_collection,
                                    state="disabled")
        self.btn_start.grid(row=0, column=2, rowspan=3,
                            padx=10, pady=5, ipady=15, sticky="e")

        # ===========================================
        # 2. 왼쪽 영역: 라벨링 버튼
        # ===========================================
        labeling_frame = ttk.LabelFrame(left_frame, text="2. 데이터 라벨링")
        labeling_frame.pack(fill="x")

        # --- 현재 처리 대상 파일 표시 ---
        ttk.Label(labeling_frame, text="현재 파일:").pack(pady=5)
        self.lbl_current_file_var = tk.StringVar(value="(없음)")
        ttk.Label(labeling_frame,
                  textvariable=self.lbl_current_file_var,
                  foreground="blue",
                  font=("", 10, "bold")).pack(pady=(0, 10))

        # --- 라벨링 버튼 묶음 ---
        button_frame = ttk.Frame(labeling_frame)
        button_frame.pack(pady=5, fill="x", expand=True)

        self.btn_discard = ttk.Button(button_frame, text="버리기",
                                      command=self.discard_file,
                                      state="disabled")
        self.btn_discard.pack(fill="x", expand=True, padx=5, pady=2)

        self.btn_normal = ttk.Button(button_frame, text="정상 (0)",
                                     command=self.label_normal,
                                     state="disabled")
        self.btn_normal.pack(fill="x", expand=True, padx=5, pady=2)

        self.btn_fall = ttk.Button(button_frame, text="넘어짐 (1)",
                                   command=self.label_fall,
                                   state="disabled")
        self.btn_fall.pack(fill="x", expand=True, padx=5, pady=2)

        self.btn_fake = ttk.Button(button_frame, text="넘어진 척 (2)",
                                   command=self.label_fake,
                                   state="disabled")
        self.btn_fake.pack(fill="x", expand=True, padx=5, pady=2)

        # ===========================================
        # 3. 오른쪽 영역: 그래프 (실시간 / 스냅샷)
        # ===========================================
        # --- 상단: 실시간 그래프 ---
        plot_frame = ttk.LabelFrame(right_frame, text="실시간 센서 (최근 7초)")
        plot_frame.pack(fill="both", expand=True, pady=(0, 5))

        self.fig1 = Figure(figsize=(7, 3.5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- 하단: 수집된 데이터 스냅샷 그래프 ---
        static_plot_frame = ttk.LabelFrame(right_frame, text="수집된 데이터 (스냅샷)")
        static_plot_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.fig2 = Figure(figsize=(7, 3.5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=static_plot_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- 그래프용 버퍼 및 기본 축 설정 ---
        self.plot_buffer = deque(
            [self.last_valid_data] * self.plot_window_size,
            maxlen=self.plot_window_size
        )
        self.plot_time_axis = np.linspace(0, PLOT_WINDOW_SEC,
                                          self.plot_window_size)
        self.init_plots()

        # --- Hz, Sec 값 변경 시 샘플 수 계산 ---
        self.hz_var.trace_add("write", self.update_sample_count)
        self.sec_var.trace_add("write", self.update_sample_count)
        self.update_sample_count()

        # --- 창 닫힐 때 정리 ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- 프로그램 시작 시 자동 시리얼 연결 ---
        self.connect_serial()

    # ===========================================
    # 그래프 축/레이블 초기화
    # ===========================================
    def init_plots(self):
        """실시간/스냅샷 그래프의 기본 축과 레이블을 설정한다."""
        # 실시간 그래프
        self.ax1.clear()
        self.ax1.set_title("Live Sensor Data")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Value")
        self.ax1.set_ylim(0, 4096)
        self.ax1.set_xlim(0, PLOT_WINDOW_SEC)
        self.ax1.grid(True)
        self.line1, = self.ax1.plot(self.plot_time_axis,
                                    self.plot_buffer, lw=1.5)
        self.fig1.tight_layout(pad=0.5)
        self.canvas1.draw()

        # 스냅샷 그래프
        self.ax2.clear()
        self.ax2.set_title("Collected Data (Snapshot)")
        self.ax2.set_ylim(0, 4096)
        self.ax2.grid(True)
        self.fig2.tight_layout(pad=0.5)
        self.canvas2.draw()

    # ===========================================
    # Hz / Sec 변경 시 샘플 수 업데이트
    # ===========================================
    def update_sample_count(self, *args):
        """Hz, Sec 입력값이 바뀔 때 총 샘플 수와 버퍼 크기를 갱신한다."""
        try:
            sample_rate = int(self.hz_var.get())
            collection_sec = float(self.sec_var.get())
            self.collection_window_size = int(sample_rate * collection_sec)
            self.lbl_sample_count.config(
                text=f"= 총 {self.collection_window_size} 샘플"
            )

            # 실시간 그래프 버퍼 크기도 Hz에 맞춰 조정
            new_plot_window_size = int(sample_rate * PLOT_WINDOW_SEC)
            if (not self.is_collecting) and self.plot_window_size != new_plot_window_size:
                self.plot_window_size = new_plot_window_size
                self.plot_buffer = deque(
                    [self.last_valid_data] * self.plot_window_size,
                    maxlen=self.plot_window_size
                )
                self.plot_time_axis = np.linspace(
                    0, PLOT_WINDOW_SEC, self.plot_window_size
                )
        except ValueError:
            self.lbl_sample_count.config(text="= 계산 오류")

    # ===========================================
    # 시리얼 포트 연결
    # ===========================================
    def connect_serial(self):
        """시리얼 포트를 열고, 실시간 그래프 애니메이션을 시작한다."""
        if self.ser and self.ser.is_open:
            return  # 이미 연결되어 있으면 무시

        self.status_var.set(f"'{SERIAL_PORT}'에 연결 시도 중...")

        try:
            self.sample_rate = int(self.hz_var.get())
            if self.sample_rate <= 0:
                raise ValueError("Hz > 0 필요")

            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.ser.flushInput()

            self.status_var.set(
                f"'{SERIAL_PORT}' 연결 성공 ({self.sample_rate} Hz). 그래프 시작."
            )

            # GUI 활성화
            self.entry_hz.config(state="normal")
            self.entry_sec.config(state="normal")
            self.btn_start.config(state="normal")
            self.update_label_buttons()

            # 그래프 초기화 및 애니메이션 시작
            self.init_plots()
            self.ani = FuncAnimation(
                self.fig1,
                self.update_plot,
                interval=ANIMATION_INTERVAL_MS,
                blit=False,
                cache_frame_data=False
            )

        except ValueError as e:
            messagebox.showerror(
                "입력 오류",
                f"샘플링 속도(Hz)는 양의 정수여야 합니다.\n{e}"
            )
            self.status_var.set("오류: Hz 값을 확인하세요.")
            self.ser = None

        except serial.SerialException as e:
            messagebox.showerror(
                "연결 오류",
                f"포트 '{SERIAL_PORT}' 연결에 실패했습니다:\n{e}"
            )
            self.status_var.set("오류: 시리얼 포트를 확인하세요.")
            self.ser = None

        except Exception as e:
            messagebox.showerror("오류", f"예상치 못한 오류:\n{e}")
            self.status_var.set("오류 발생.")
            self.ser = None

    # ===========================================
    # 시리얼 포트 해제
    # ===========================================
    def disconnect_serial(self):
        """시리얼 포트와 그래프 애니메이션을 종료한다."""
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None

        if self.ser and self.ser.is_open:
            self.ser.close()

        self.ser = None
        self.status_var.set("연결 해제됨.")
        self.update_label_buttons()
        self.btn_start.config(state="disabled")
        self.entry_hz.config(state="disabled")
        self.entry_sec.config(state="disabled")

    # ===========================================
    # 창 닫기 이벤트
    # ===========================================
    def on_closing(self):
        """창이 닫힐 때 시리얼 포트 정리 후 종료."""
        self.disconnect_serial()
        self.root.destroy()

    # ===========================================
    # 센서 데이터 읽기
    # ===========================================
    def get_sensor_data(self):
        """
        시리얼 포트에서 한 줄을 읽어 정수로 변환한다.
        읽기 실패 시 마지막 정상값을 그대로 사용한다.
        """
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

    # ===========================================
    # 실시간 그래프 업데이트 (애니메이션 콜백)
    # ===========================================
    def update_plot(self, frame):
        """주기적으로 호출되어 실시간 그래프를 한 스텝 이동시키는 함수."""
        new_data = self.get_sensor_data()

        # 새 데이터 추가 (deque가 자동으로 맨 앞 요소를 밀어냄)
        self.plot_buffer.append(new_data)

        # y 데이터 갱신
        self.line1.set_ydata(self.plot_buffer)

        # 수집 중이면 버퍼에 쌓기
        if self.is_collecting:
            self.collection_buffer.append(new_data)
            self.collection_step_counter += 1

            if self.collection_step_counter >= self.collection_window_size:
                self.stop_collection()

        return self.line1,

    # ===========================================
    # 수집 시작
    # ===========================================
    def start_collection(self):
        """수집 버튼을 눌렀을 때 한 번의 데이터 수집을 시작한다."""
        if self.is_collecting:
            return

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

        # 시작 알림 소리
        self.root.bell()

        # 상태 플래그 및 버퍼 초기화
        self.is_collecting = True
        self.collection_buffer.clear()
        self.collection_step_counter = 0

        # 수집 중에는 라벨링 버튼 비활성화
        self.update_label_buttons()

        self.status_var.set(f"수집 중... ({self.collection_window_size} 샘플)")

        # 스냅샷 그래프를 "Collecting..."으로 초기화
        self.ax2.clear()
        self.ax2.set_title("Collecting...")
        self.ax2.set_ylim(0, 4096)
        self.ax2.grid(True)
        self.canvas2.draw()

    # ===========================================
    # 수집 종료
    # ===========================================
    def stop_collection(self):
        """수집이 끝났을 때 스냅샷 표시 및 RAW 파일 저장."""
        # 종료 알림 소리
        self.root.bell()

        self.is_collecting = False
        collected_data = list(self.collection_buffer)

        # --- 스냅샷 그래프 그리기 ---
        self.ax2.clear()
        self.ax2.set_title(f"Collected ({len(collected_data)} samples)")
        self.ax2.set_ylim(0, 4096)
        self.ax2.grid(True)
        self.ax2.plot(collected_data)
        self.fig2.tight_layout(pad=0.5)
        self.canvas2.draw()

        # --- RAW 데이터 csv로 저장 ---
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}.csv"
            filepath = os.path.join(RAW_DATA_FOLDER, filename)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(collected_data)

            self.current_file_path = filepath
            self.lbl_current_file_var.set(filename)
            self.status_var.set("수집 완료! 라벨링 대기 중.")
        except Exception as e:
            messagebox.showerror("파일 저장 오류", str(e))
            self.status_var.set("파일 저장 중 오류 발생.")
            self.current_file_path = None
            self.lbl_current_file_var.set("(없음)")

        # 수집 종료 후 라벨 버튼 다시 활성화 여부 갱신
        self.update_label_buttons()

    # ===========================================
    # 라벨 버튼 활성/비활성 처리
    # ===========================================
    def update_label_buttons(self):
        """
        현재 연결/수집/파일 상태에 따라
        라벨링 버튼과 입력창/시작 버튼의 활성/비활성을 조정한다.
        """
        is_connected = self.ser is not None and self.ser.is_open
        has_file = self.current_file_path is not None

        # 수집 중이면 시작 버튼/입력창 비활성화
        if is_connected and not self.is_collecting:
            self.btn_start.config(state="normal")
            self.entry_hz.config(state="normal")
            self.entry_sec.config(state="normal")
        else:
            self.btn_start.config(state="disabled")
            self.entry_hz.config(state="disabled")
            self.entry_sec.config(state="disabled")

        # 라벨 버튼: 연결되어 있고, 수집 중이 아니고, 파일이 있을 때만 활성화
        label_state = "normal" if (is_connected and not self.is_collecting and has_file) else "disabled"
        self.btn_discard.config(state=label_state)
        self.btn_normal.config(state=label_state)
        self.btn_fall.config(state=label_state)
        self.btn_fake.config(state=label_state)

    # ===========================================
    # 공통 파일 처리 로직 (라벨링/삭제)
    # ===========================================
    def process_file(self, label):
        """
        현재 파일을 라벨에 따라:
        - 삭제(-1)
        - 정상(0) / 넘어짐(1) / 넘어진 척(2) 폴더로 이동하며,
          CSV 마지막 칸에 라벨을 추가해서 저장한다.
        """
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            messagebox.showwarning("오류", "처리할 파일이 없습니다.")
            return

        original_filename = os.path.basename(self.current_file_path)

        try:
            # --- 삭제 라벨 ---
            if label == -1:
                os.remove(self.current_file_path)
                status_message = f"파일 '{original_filename}' 삭제됨."

            # --- 분류 라벨 ---
            else:
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
                elif label == 2:
                    self.fake_count += 1
                    count = self.fake_count
                    dest_folder = FAKE_FOLDER
                    prefix = "fake"
                else:
                    return  # 정의되지 않은 라벨이면 무시

                new_filename = f"{prefix}_{count}번째_{original_filename}"
                dest_path = os.path.join(dest_folder, new_filename)

                # 원본 csv 읽기
                with open(self.current_file_path, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    data_row = next(reader)

                # 라벨 추가
                data_row.append(label)

                # 새 파일로 쓰기
                with open(dest_path, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(data_row)

                # RAW 파일 삭제
                os.remove(self.current_file_path)
                status_message = f"-> '{new_filename}'(으)로 저장됨."

            # 공통 GUI 갱신
            self.current_file_path = None
            self.lbl_current_file_var.set("(없음)")
            self.status_var.set(status_message)
            self.update_label_buttons()

        except Exception as e:
            messagebox.showerror("파일 처리 오류", str(e))
            self.status_var.set("파일 처리 중 오류 발생.")

    # ===========================================
    # 라벨링 버튼 콜백
    # ===========================================
    def discard_file(self):
        """버리기 버튼: RAW 파일 삭제"""
        self.process_file(label=-1)

    def label_normal(self):
        """정상 (0) 버튼"""
        self.process_file(label=0)

    def label_fall(self):
        """넘어짐 (1) 버튼"""
        self.process_file(label=1)

    def label_fake(self):
        """넘어진 척 (2) 버튼"""
        self.process_file(label=2)


# ===========================================
# 메인 진입점
# ===========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DataLabelingApp(root)
    root.mainloop()