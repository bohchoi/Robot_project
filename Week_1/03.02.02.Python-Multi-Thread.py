import threading
import time
import random

robot_telemetry = {"lidar": 0, "ultrasonic": 0}
data_lock = threading.Lock()

def read_lidar():
    while True:
        distance = random.uniform(0.1, 10.0)
        with data_lock:
            robot_telemetry["lidar"] = distance
        time.sleep(0.1)  # 10Hz

def read_ultrasonic():
    while True:
        time.sleep(0.5)
        distance = random.uniform(20, 400)
        with data_lock:
            robot_telemetry["ultrasonic"] = distance

if __name__ == "__main__":
    # 데몬 스레드: 메인 종료 시 함께 종료
    t1 = threading.Thread(target=read_lidar, daemon=True)
    t2 = threading.Thread(target=read_ultrasonic, daemon=True)

    t1.start()
    t2.start()

    try:
        for _ in range(3):  # 3회만 출력 후 종료 (실습 편의상)
            with data_lock:
                print(f"[Main Control] Lidar: {robot_telemetry['lidar']:.2f}m, "
                      f"Ultrasonic: {robot_telemetry['ultrasonic']:.2f}cm")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n로봇 제어 프로그램을 종료합니다.")

    print("메인 루프 종료 → 데몬 스레드 자동 종료")