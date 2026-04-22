# sudo apt install python3-tk
import matplotlib
matplotlib.use('TkAgg')   # 또는 'Qt5Agg'
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import json


# 센서 추상 클래스: 모든 센서의 설계도 역할
class Sensor(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def get_data(self):
        """센서로부터 데이터를 읽어오는 추상 메서드"""
        pass


# LiDAR 센서 클래스: 거리 데이터를 생성
class LidarSensor(Sensor):
    def __init__(self, name, n_sample = 360):
        super().__init__(name)
        self.n_sample = n_sample
    
    def get_data(self):
        angles = np.linspace(0, np.pi * 2, self.n_sample)
        distances = 5.0 + np.random.normal(0, 0.1, self.n_sample)
        return angles, distances


# 실시간 LiDAR 스캔 시뮬레이션 (10회 반복)
lidar_rt = LidarSensor("Hokuyo_LiDAR")

plt.ion()  # 인터랙티브 모드 활성화
fig_rt, ax_rt = plt.subplots(figsize=(6, 6))

for frame in range(1000):  # 무한 루프 대신 10 프레임만 반복
    ax_rt.clear()
    angles, dists = lidar_rt.get_data()
    x = dists * np.cos(angles)
    y = dists * np.sin(angles)
    ax_rt.scatter(x, y, s=5, c='blue')
    ax_rt.set_title(f"LiDAR Real-time Scan (frame {frame + 1}/10)")
    ax_rt.set_aspect('equal')
    ax_rt.set_xlim(-7, 7)
    ax_rt.set_ylim(-7, 7)
    plt.pause(0.1)  # 100ms 대기 후 업데이트

plt.ioff()  # 인터랙티브 모드 해제
plt.show()