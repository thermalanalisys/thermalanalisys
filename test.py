import numpy as np
import matplotlib.pyplot as plt

# 메쉬수, 시간
size = 101
t = 1500

# 초기온도분포
T0 = np.zeros((size))
# 체적당 비열 분포
crho = np.ones((size))
# 위치별 열전도율
k = np.ones((size))
# 계산을 위한 인접한 노드간 평균열전도율
k_x = np.ones((size - 1))
# 그래디언트를 만드는 행렬
grad = np.zeros((size - 1, size))
# 다이버전스를 만드는 행렬
div = np.zeros((size, size - 1))

for i in range(0, size):
    T0[i] += i

for i in range(0, size):
    k[i] = 1 + (i / size)

for i in range(0, size - 1):
    k_x[i] = (k[i] + k[i + 1]) / 2

for i in range(0, size):
    crho[i] = 1

A = np.zeros((size, size))  # 열방정식 우변

for i in range(0, size - 1):
    grad[i][i] = -1
    grad[i][i + 1] = 1
    div[i][i] = 1
    div[i + 1][i] = -1

A = np.dot(div, np.dot(np.diag(k_x), grad))

for i in range(0, size):
    for j in range(0, size):
        A[i][j] /= crho[i]

# 경계조건
# for i in range(0,size):
# A[round(size/2)] = 0


# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

# 고유벡터 행렬ㅡㅡㅡ P와 그 역행렬 P^-1 계산
P = eigenvectors
P_inv = np.linalg.inv(P)

# 고유값의 제곱근을 대각행렬로 변환
D = np.diag(eigenvalues)
exp_D = np.diag(np.exp(eigenvalues * t))

result = np.dot(np.dot(P, exp_D), P_inv)

# 결과 출력
print("원래 행렬 A:")
print(A)
y = np.dot(result, T0)
x = [i for i in range(0, size)]

# 온도 필드 시각화
plt.plot(x, y)
plt.show()

avg = 0
crho_sum = 0
for i in range(0, size):
    avg += crho[i] * np.dot(result, T0)[i]
    crho_sum += crho[i]

avg /= crho_sum
print("평균온도: ", avg)
print('\n')
print("고윳값:")
print('\n')
print(eigenvalues)
