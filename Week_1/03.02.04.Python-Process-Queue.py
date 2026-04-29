import multiprocessing
import time
import random

# 1. 센서 데이터를 읽는 워커 프로세스
def sensor_worker(sensor_queue, stop_event):
    print("센서 프로세스 시작")
    while not stop_event.is_set():
        data = random.uniform(0, 100)
        sensor_queue.put(data)
        time.sleep(0.01)  # 100Hz 속도로 읽기
    print("센서 프로세스 종료")

# 2. AI 추론을 수행하는 워커 프로세스
def ai_inference_worker(sensor_queue, stop_event):
    print("AI 추론 프로세스 시작")
    while not stop_event.is_set():
        try:
            data = sensor_queue.get(timeout=1)
            print(f"[AI] 데이터 {data:.2f} 분석 중...")
            time.sleep(0.2)  # 무거운 AI 연산 시뮬레이션
            print(f"[AI] 결과: 장애물 없음")
        except:
            continue
    print("AI 추론 프로세스 종료")

if __name__ == '__main__':
    # 프로세스 간 통신: Queue, 신호 전달: Event
    sensor_q = multiprocessing.Queue()
    stop_sig = multiprocessing.Event()

    # 프로세스 객체 생성
    p1 = multiprocessing.Process(
            target=sensor_worker, args=(sensor_q, stop_sig))
    p2 = multiprocessing.Process(
            target=ai_inference_worker, args=(sensor_q, stop_sig))

    p1.start()
    p2.start()

    try:
        time.sleep(2)  # 2초 동안 실행 (실습 편의상 단축)
    except KeyboardInterrupt:
        pass
    finally:
        stop_sig.set()   # 종료 신호 전송
        p1.join()
        p2.join()
        print("모든 시스템이 안전하게 종료되었습니다.")