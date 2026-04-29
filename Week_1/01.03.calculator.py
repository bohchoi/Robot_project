# 5단계: try-except를 이용한 견고한 코드 만들기

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다.")
    return a / b


def calculator():
    """로봇 SW 개발 입문용 계산기 메인 함수"""
    print("계산기 시작 (종료: q)")
    while True:
        try:
            op = input("\n연산자 (+, -, *, /, q): ")
            if op == 'q':
                print("계산기를 종료합니다.")
                break
            if op not in ['+', '-', '*', '/']:
                continue

            num1 = float(input("첫 번째 숫자: "))
            num2 = float(input("두 번째 숫자: "))

            if op == '+':
                result = add(num1, num2)
            elif op == '-':
                result = sub(num1, num2)
            elif op == '*':
                result = mul(num1, num2)
            elif op == '/':
                result = div(num1, num2)
            else:
                print(f"오류: '{op}'는 지원하지 않는 연산자입니다.")
                continue

            print(f"{num1} {op} {num2} = {result}")

        except ValueError as e:
            print(f"입력 오류: {e}")   # 숫자 아닌 입력 또는 0 나누기
        except Exception as e:
            print(f"알 수 없는 오류: {e}")


if __name__ == "__main__":
    calculator()