import threading
import time

def heartbeat():
    while True:
        print("Robot alive...")
        time.sleep(0.1)

# daemon=True 로 생성 시 설정
t = threading.Thread(target=heartbeat, daemon=True)
t.start()

# 메인 프로그램이 0.35초 후 종료되면 데몬 스레드도 함께 종료됨
time.sleep(0.35)
print("메인 프로그램 종료 → 데몬 스레드 자동 종료")