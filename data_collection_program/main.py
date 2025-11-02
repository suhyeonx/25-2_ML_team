import tkinter as tk
from tkinter import ttk, messagebox, font
import serial
import serial.tools.list_ports
import threading
import queue
import collections
import time
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# import matplotlib.animation as animation  # <- 제거 (더 이상 사용 안 함)

# --- ⚙️ 전역 상수 설정 ---
LIVE_GRAPH_SECONDS = 7
EXPECTED_HZ = 50
LIVE_BUFFER_MAXLEN = LIVE_GRAPH_SECONDS * EXPECTED_HZ
DATA_SAVE_BASE_DIR = "data"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1
GUI_UPDATE_INTERVAL_MS = 100  # GUI 업데이트 주기 (100ms = 10Hz)


class SerialDataLoggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("데이터 수집 프로그램(v2.0.0)")
        self.root.geometry("1200x700")

        # --- 데이터 버퍼 및 동기화 객체 초기화 ---
        self.data_lock = threading.Lock()
        self.live_data_buffer = collections.deque(maxlen=LIVE_BUFFER_MAXLEN)
        self.recorded_data_buffer = []

        # --- 스레드 제어용 이벤트 ---
        self.is_monitoring_event = threading.Event()
        self.is_collecting_event = threading.Event()

        # --- 스레드 간 통신 큐 ---
        self.worker_to_gui_queue = queue.Queue()

        # --- 스레드 및 시리얼 객체 ---
        self.serial_connection = None
        self.worker_thread = None

        # --- 상태 변수 ---
        self.target_samples_to_collect = EXPECTED_HZ * 6
        self.collection_start_timestamp = None
        self.collection_end_timestamp = None

        # --- GUI 생성 ---
        self._create_widgets()

        # --- ⭐️ GUI 큐 폴링 및 그래프 업데이트 시작 ---
        # FuncAnimation 대신 root.after 루프가 모든 것을 처리합니다.
        self.root.after(GUI_UPDATE_INTERVAL_MS, self._gui_update_loop)

        # --- 창 닫기 이벤트 바인딩 ---
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ------------------------------------------------------------------
    # 1. GUI 위젯 생성 (변경 없음)
    # ------------------------------------------------------------------
    def _create_widgets(self):
        # (이 함수 내용은 이전과 동일합니다)
        # --- 메인 프레임 (좌: 컨트롤, 우: 그래프) ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1, minsize=300)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_rowconfigure(0, weight=1)

        # --- 1-1. 왼쪽 컨트롤 패널 ---
        control_panel = ttk.Frame(main_frame)
        control_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        control_panel.grid_rowconfigure(1, weight=1)
        control_panel.grid_columnconfigure(0, weight=1)

        # --- "설정" LabelFrame ---
        settings_frame = ttk.LabelFrame(control_panel, text="설정", padding="10")
        settings_frame.grid(row=0, column=0, sticky="new")
        settings_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="연결 포트:").grid(row=0, column=0, sticky="w", pady=5)
        self.port_combobox = ttk.Combobox(settings_frame, state="readonly")
        self.port_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.port_combobox.bind("<Button-1>", self._refresh_ports_list)
        self.port_combobox.bind("<<ComboboxSelected>>", self._on_port_select)

        ttk.Label(settings_frame, text="샘플링 속도:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Label(settings_frame, text=f"{EXPECTED_HZ} Hz (고정)").grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(settings_frame, text="수집 시간 (초):").grid(row=2, column=0, sticky="w", pady=5)
        self.collect_time_var = tk.StringVar(value="6")
        self.collect_time_entry = ttk.Entry(settings_frame, textvariable=self.collect_time_var, width=10)
        self.collect_time_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.collect_time_entry.bind("<KeyRelease>", self._update_total_samples_label)

        self.total_samples_label = ttk.Label(settings_frame, text=f"총 샘플: {self.target_samples_to_collect}개")
        self.total_samples_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        self.collect_button = ttk.Button(settings_frame, text="수집 시작", command=self._on_collect_start,
                                         state=tk.DISABLED)
        self.collect_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10, padx=5)

        # --- "라벨링 및 저장" LabelFrame ---
        labeling_frame = ttk.LabelFrame(control_panel, text="라벨링 및 저장", padding="10")
        labeling_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        labeling_frame.grid_columnconfigure(0, weight=1)
        labeling_frame.grid_columnconfigure(1, weight=1)

        self.status_label = ttk.Label(labeling_frame, text="상태: 포트 선택 대기 중...", font=font.Font(weight='bold'))
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        self.label_buttons = []
        btn_texts = ["Type 1. 정상", "Type 2. 낙상", "Type 3. 어려운 정상"]
        for i, text in enumerate(btn_texts):
            btn = ttk.Button(labeling_frame, text=text, command=lambda t=i + 1: self._save_data(t), state=tk.DISABLED)
            btn.grid(row=i + 1, column=0, sticky="ew", padx=5, pady=3)
            self.label_buttons.append(btn)

        self.discard_button = ttk.Button(labeling_frame, text="버리기", command=self._discard_data, state=tk.DISABLED)
        self.discard_button.grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        self.label_buttons.append(self.discard_button)

        self.save_status_label = ttk.Label(labeling_frame, text="", foreground="blue")
        self.save_status_label.grid(row=len(btn_texts) + 1, column=0, columnspan=2, sticky="w", pady=10, padx=5)

        # --- 1-2. 오른쪽 그래프 패널 ---
        graph_frame = ttk.Frame(main_frame)
        graph_frame.grid(row=0, column=1, sticky="nsew")
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        plt.style.use('ggplot')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False, tight_layout=True)

        # Ax1: 실시간 모니터
        self.ax1.set_title(f"Real-time Monitor (Last {LIVE_GRAPH_SECONDS} sec)")
        self.ax1.set_ylabel("Sensor Value")
        self.line1, = self.ax1.plot([], [], 'b-')
        self.ax1.set_xlim(-LIVE_GRAPH_SECONDS, 0)
        self.ax1.set_ylim(0, 1024)
        self.start_line_marker = self.ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, visible=False)
        self.end_line_marker = self.ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, visible=False)

        # Ax2: 녹화 데이터
        self.ax2.set_title("Recorded Data")
        self.ax2.set_xlabel("Time (sec)")
        self.ax2.set_ylabel("Sensor Value")
        self.line2, = self.ax2.plot([], [], 'r-')
        self.ax2.set_ylim(0, 1024)
        self.ax2_text = self.ax2.text(0.5, 0.5, "Waiting for data collection",
                                      horizontalalignment='center', verticalalignment='center',
                                      transform=self.ax2.transAxes, fontsize=14, color='gray')

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Matplotlib 애니메이션 시작 (제거됨) ---
        # self.ani = animation.FuncAnimation(...) <- 이 줄이 삭제됨

        # 초기 포트 목록 로드
        self._refresh_ports_list()

    # ------------------------------------------------------------------
    # 2. GUI 콜백 및 헬퍼 함수 (이전과 동일)
    # ------------------------------------------------------------------
    def _refresh_ports_list(self, event=None):
        ports = serial.tools.list_ports.comports()
        port_names = [p.device for p in ports]
        self.port_combobox['values'] = port_names
        if port_names and not self.serial_connection:
            self.port_combobox.set(port_names[0])
        print("포트 목록 새로고침:", port_names)

    def _update_total_samples_label(self, event=None):
        try:
            seconds = int(self.collect_time_var.get())
            if seconds <= 0: raise ValueError
            self.target_samples_to_collect = EXPECTED_HZ * seconds
            self.total_samples_label.config(text=f"총 샘플: {self.target_samples_to_collect}개")
        except ValueError:
            self.total_samples_label.config(text="총 샘플: (유효한 시간 입력)")

    def _on_port_select(self, event=None):
        selected_port = self.port_combobox.get()
        if not selected_port:
            return
        print(f"{selected_port} 선택, 연결 시도...")
        self._stop_worker_thread()
        try:
            self.serial_connection = serial.Serial(selected_port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
            print(f"시리얼 연결 성공: {selected_port}")
            self.is_monitoring_event.set()
            self.worker_thread = threading.Thread(target=self._serial_reader_thread, daemon=True)
            self.worker_thread.start()
            self.status_label.config(text=f"상태: {selected_port} 모니터링 중...")
            self.collect_button.config(state=tk.NORMAL)
            self.save_status_label.config(text="")
        except serial.SerialException as e:
            messagebox.showerror("연결 오류", f"{selected_port}에 연결할 수 없습니다.\n{e}")
            self.status_label.config(text="상태: 연결 실패")
            self.serial_connection = None

    def _on_collect_start(self):
        if not self.serial_connection or not self.serial_connection.is_open:
            messagebox.showwarning("오류", "시리얼 포트가 연결되지 않았습니다.")
            return
        self.root.bell()
        with self.data_lock:
            self.recorded_data_buffer.clear()
        self.collection_start_timestamp = time.time()
        self.collection_end_timestamp = None
        self.is_collecting_event.set()
        self.collect_button.config(state=tk.DISABLED, text="수집 중...")
        self.status_label.config(text="상태: 데이터 수집 중...")
        self.save_status_label.config(text="")
        self._set_label_buttons_state(tk.DISABLED)
        self.ax2_text.set_visible(True)
        self.ax2_text.set_text(f"Collecting... (0 / {self.target_samples_to_collect})")
        self.line2.set_data([], [])
        # self.canvas.draw_idle() # <- 여기서 그릴 필요 없음. _gui_update_loop가 그려줌

    def _on_collection_finished(self):
        self.root.bell()
        self.collection_end_timestamp = time.time()
        with self.data_lock:
            recorded_data_copy = list(self.recorded_data_buffer)
        collected_count = len(recorded_data_copy)
        print(f"수집 완료. 총 {collected_count}개 샘플 수집됨.")
        self.collect_button.config(state=tk.NORMAL, text="수집 시작")
        if collected_count > 0:
            self.status_label.config(text=f"상태: 라벨링 대기 중 ({collected_count}개 샘플)")
            self._set_label_buttons_state(tk.NORMAL)
            timestamps, values = zip(*recorded_data_copy)
            relative_time = [ts - timestamps[0] for ts in timestamps]
            self.ax2_text.set_visible(False)
            self.line2.set_data(relative_time, values)
            self.ax2.set_xlim(0, relative_time[-1])
            self.ax2.set_title(f"Recorded Data ({collected_count} samples)")
        else:
            self.status_label.config(text="상태: 수집된 데이터 없음.")
            self.ax2_text.set_visible(True)
            self.ax2_text.set_text("No data collected")
        # self.canvas.draw_idle() # <- 여기서 그릴 필요 없음. _gui_update_loop가 그려줌

    def _save_data(self, label_type):
        with self.data_lock:
            data_to_save = list(self.recorded_data_buffer)
            self.recorded_data_buffer.clear()
        if not data_to_save:
            self.save_status_label.config(text="저장할 데이터가 없습니다.", foreground="red")
            return
        try:
            save_dir = os.path.join(DATA_SAVE_BASE_DIR, f"type{label_type}")
            os.makedirs(save_dir, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"type{label_type}_{timestamp_str}.csv"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "value"])
                writer.writerows(data_to_save)
            print(f"데이터 저장 완료: {filepath}")
            self.save_status_label.config(text=f"저장 완료: {filename}", foreground="blue")
        except Exception as e:
            print(f"데이터 저장 실패: {e}")
            messagebox.showerror("저장 오류", f"데이터 저장에 실패했습니다:\n{e}")
            self.save_status_label.config(text="저장 실패.", foreground="red")
        self._set_label_buttons_state(tk.DISABLED)
        self.status_label.config(text="상태: 모니터링 중...")
        self.line2.set_data([], [])
        self.ax2_text.set_visible(True)
        self.ax2_text.set_text("Waiting for data collection")
        self.ax2.set_title("Recorded Data")
        # self.canvas.draw_idle() # <- 여기서 그릴 필요 없음. _gui_update_loop가 그려줌

    def _discard_data(self):
        with self.data_lock:
            self.recorded_data_buffer.clear()
        print("녹화된 데이터 버림.")
        self.save_status_label.config(text="데이터가 버려졌습니다.", foreground="orange")
        self._set_label_buttons_state(tk.DISABLED)
        self.status_label.config(text="상태: 모니터링 중...")
        self.line2.set_data([], [])
        self.ax2_text.set_visible(True)
        self.ax2_text.set_text("Waiting for data collection")
        self.ax2.set_title("Recorded Data")
        # self.canvas.draw_idle() # <- 여기서 그릴 필요 없음. _gui_update_loop가 그려줌

    def _set_label_buttons_state(self, state):
        for btn in self.label_buttons:
            btn.config(state=state)

    # ------------------------------------------------------------------
    # 3. ⭐️ 그래프 업데이트 (메인 스레드, FuncAnimation 대신)
    # ------------------------------------------------------------------
    def _update_graphs(self):
        """(메인 스레드) 실시간 그래프(ax1)를 업데이트합니다. (frame 인자 제거)"""

        with self.data_lock:
            live_data_copy = list(self.live_data_buffer)

        if not live_data_copy:
            return  # 데이터 없으면 그릴 필요 없음

        # --- 실시간 그래프(ax1) 업데이트 ---
        timestamps, values = zip(*live_data_copy)
        current_time = time.time()
        relative_time = [ts - current_time for ts in timestamps]

        self.line1.set_data(relative_time, values)

        min_val = min(values) - 10
        max_val = max(values) + 10
        self.ax1.set_ylim(min_val, max_val)  # Y축 자동 스케일링

        # --- 수집 중/종료 세로줄 업데이트 ---
        if self.collection_start_timestamp:
            start_rel_time = self.collection_start_timestamp - current_time
            if -LIVE_GRAPH_SECONDS < start_rel_time < 0:
                self.start_line_marker.set_xdata([start_rel_time, start_rel_time])
                self.start_line_marker.set_visible(True)
            else:
                self.start_line_marker.set_visible(False)
                if start_rel_time < -LIVE_GRAPH_SECONDS:
                    self.collection_start_timestamp = None
        else:
            self.start_line_marker.set_visible(False)

        if self.collection_end_timestamp:
            end_rel_time = self.collection_end_timestamp - current_time
            if -LIVE_GRAPH_SECONDS < end_rel_time < 0:
                self.end_line_marker.set_xdata([end_rel_time, end_rel_time])
                self.end_line_marker.set_visible(True)
            else:
                self.end_line_marker.set_visible(False)
                if end_rel_time < -LIVE_GRAPH_SECONDS:
                    self.collection_end_timestamp = None
        else:
            self.end_line_marker.set_visible(False)

        # 수집 중 텍스트 업데이트 (ax2)
        if self.is_collecting_event.is_set():
            with self.data_lock:
                count = len(self.recorded_data_buffer)
            self.ax2_text.set_text(f"Collecting... ({count} / {self.target_samples_to_collect})")

        # ⭐️ FuncAnimation의 return 대신, 여기서 직접 캔버스 그리기를 요청합니다.
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 4. 워커 스레드 (이전과 동일)
    # ------------------------------------------------------------------
    def _serial_reader_thread(self):
        """(워커 스레드) 시리얼 데이터를 읽고 버퍼에 채웁니다."""
        print("워커 스레드 시작...")

        try:
            while self.is_monitoring_event.is_set():
                if not self.serial_connection or not self.serial_connection.is_open:
                    break

                try:
                    line = self.serial_connection.readline()
                    if not line:
                        continue

                    data_str = line.decode('utf-8', errors='ignore').strip()
                    if not data_str:
                        continue

                    value_str = data_str.split(',')[0]
                    value = float(value_str)
                    timestamp = time.time()

                    with self.data_lock:
                        self.live_data_buffer.append((timestamp, value))

                        if self.is_collecting_event.is_set():
                            self.recorded_data_buffer.append((timestamp, value))

                            if len(self.recorded_data_buffer) >= self.target_samples_to_collect:
                                self.is_collecting_event.clear()
                                self.worker_to_gui_queue.put("COLLECTION_DONE")

                except ValueError:
                    print(f"데이터 파싱 오류: '{data_str}'")
                except serial.SerialException as e:
                    print(f"시리얼 오류: {e}")
                    self.worker_to_gui_queue.put("DISCONNECTED")
                    break
                except Exception as e:
                    print(f"워커 스레드 예외: {e}")
                    time.sleep(0.01)

        finally:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            print("워커 스레드 종료 및 포트 닫힘.")

    # ------------------------------------------------------------------
    # 5. ⭐️ 큐/그래프 통합 루프 및 종료 핸들러 (수정됨)
    # ------------------------------------------------------------------
    def _gui_update_loop(self):
        """(메인 스레드) 큐를 처리하고 그래프를 업데이트하는 통합 루프."""
        try:
            # 1. 큐 처리
            while not self.worker_to_gui_queue.empty():
                message = self.worker_to_gui_queue.get_nowait()

                if message == "COLLECTION_DONE":
                    self._on_collection_finished()

                elif message == "DISCONNECTED":
                    messagebox.showerror("연결 끊김", "시리얼 장치 연결이 끊어졌습니다.")
                    self._stop_worker_thread()
                    self.status_label.config(text="상태: 연결 끊김")
                    self.collect_button.config(state=tk.DISABLED)

            # 2. 그래프 업데이트 (모니터링 중일 때만)
            if self.is_monitoring_event.is_set():
                self._update_graphs()

        finally:
            # 다음 업데이트 예약
            self.root.after(GUI_UPDATE_INTERVAL_MS, self._gui_update_loop)

    def _stop_worker_thread(self):
        """워커 스레드를 안전하게 중지시킵니다."""
        if self.worker_thread and self.worker_thread.is_alive():
            print("워커 스레드 중지 요청...")
            self.is_monitoring_event.clear()
            self.is_collecting_event.clear()
            self.worker_thread.join(timeout=1.0)
            if self.worker_thread.is_alive():
                print("워커 스레드가 1초 내에 종료되지 않았습니다.")

        self.worker_thread = None
        self.is_monitoring_event.clear()
        self.is_collecting_event.clear()

        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("시리얼 연결 닫힘.")
        self.serial_connection = None

    def _on_closing(self):
        """창을 닫을 때 모든 리소스를 정리합니다."""
        print("애플리케이션 종료 중...")
        if messagebox.askokcancel("종료", "데이터 로거를 종료하시겠습니까?"):
            self._stop_worker_thread()
            # ⭐️ FuncAnimation 중지 코드가 필요 없어짐
            # if hasattr(self, 'ani'):
            #     self.ani.event_source.stop()
            self.root.quit()
            self.root.destroy()


# ---  Main 실행 ---
if __name__ == "__main__":
    # ⭐️ (macOS) Matplotlib 백엔드 설정 (충돌 방지를 위해 TkAgg 명시)
    # 이것이 GIL 충돌을 막는 데 도움이 될 수 있습니다.
    import matplotlib

    matplotlib.use('TkAgg')

    root = tk.Tk()
    app = SerialDataLoggerApp(root)
    root.mainloop()