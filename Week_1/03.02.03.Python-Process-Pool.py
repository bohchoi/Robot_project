from multiprocessing import Pool

def process_sensor_data(data):
    # 복잡한 필터링·계산 수행
    return data * 2

if __name__ == "__main__":
    sensor_readings = [1.2, 3.4, 5.6, 7.8, 9.0]

    with Pool(processes=4) as pool:
        results = pool.map(process_sensor_data, sensor_readings)
    print(results)  # [2.4, 6.8, 11.2, 15.6, 18.0]