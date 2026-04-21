# ---
# ## 도전 실습: 멀티스레딩 + Queue를 활용한 센서 파이프라인
#
# 아래 요구사항을 만족하는 멀티스레딩 기반 센서 파이프라인을 구현하세요.
#
# **요구사항**
# - 생산자(producer) 스레드: 0.1초마다 랜덤 센서 값을 생성하여 `queue.Queue`에 넣기
# - 소비자(consumer) 스레드: Queue에서 값을 꺼내 처리 결과 출력
# - 메인 스레드: 3초 후 두 스레드를 안전하게 종료
#
# > **힌트**: `queue.Queue`는 내부적으로 Lock이 구현되어 있어 별도 Lock 없이 Thread-safe하게 사용 가능합니다.

# %%
import threading
import queue
import time
import random

# TODO: 여기에 구현하세요

def producer(q, stop_event):
    """0.1초마다 랜덤 센서 값을 생성하여 Queue에 넣는 스레드"""
    # TODO: 구현하세요
    pass

def consumer(q, stop_event):
    """Queue에서 값을 꺼내 처리 결과를 출력하는 스레드"""
    # TODO: 구현하세요
    pass

if __name__ == "__main__":
    sensor_queue = queue.Queue()
    stop_event = threading.Event()

    t_producer = threading.Thread(target=producer, args=(sensor_queue, stop_event), daemon=True)
    t_consumer = threading.Thread(target=consumer, args=(sensor_queue, stop_event), daemon=True)

    t_producer.start()
    t_consumer.start()

    time.sleep(3)
    stop_event.set()

    t_producer.join()
    t_consumer.join()
    print("파이프라인 종료")