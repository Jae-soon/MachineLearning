import numpy as np
import matplotlib.pyplot as plt
from . import sigmoid
import time

# insert Data
np.random.seed(seed=1)
N = 200
K = 3
t = np.zeros((N, 3), dtype=np.uint8)
x = np.zeros((N, 2))
x_range0 = [-3, 3]
x_range1 = [-3, 3]
mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])
sig = np.array([[.7, .7], [.8, .3], [.3, .8]])
pi = np.array([0.4, 0.8, 1])
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < pi[k]:
            t[n, k] = 1
            break
    for k in range(2):
        x[n, k] = np.random.randn() * sig[t[n, :] == 1, k] + mu[t[n, :] == 1, k]

# 테스트, 훈련 데이터 분할
testRatio = 0.5
x_n_training = int(N * testRatio)
x_train = x[:x_n_training, :]
x_test = x[x_n_training:, :]
t_train = t[:x_n_training, :]
t_test = t[x_n_training:, :]

# 데이터 출력
def show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], linestyle='none', marker='o', markeredgecolor='black', color=c[i], alpha=0.8)
        plt.grid(True)

# 데이터 처리 네트워크 함수
def FNN(wv, M, K, x):
    N, D = x.shape # 입력차원
    w = wv[:M * (D + 1)] # 중간층 뉴런의 가중치
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):] # 출력층 뉴런의 가중치
    v = v.reshape((K, M + 1))
    b = np.zeros((N, M + 1)) # 중간층 뉴런의 입력 총합
    z = np.zeros((N, M + 1)) # 중간층 뉴런의 출력
    a = np.zeros((N, K)) # 출력층 뉴런의 입력 총합
    y = np.zeros((N, K)) # 출력층 뉴런의 출력
    for n in range(N):
        # 중간층 계산
        for m in range(M):
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1])
            z[n, m] = sigmoid(b[n, m])

        # 출력층 계산
        z[n, M] = 1 # 더미 뉴런
        wkz = 0
        for k in range(K):
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])
        for k in range(K):
            y[n, k] = np.exp(a[n, k]) / wkz
    return y, a, z, b

# 평균 교차 엔트로피 오차
def CE_FNN(wv, m, k, x, t):
    n, d = x.shape
    y, a, z, b = FNN(wv, m, k, x)
    ce = -np.dot(np.log(y.reshape(-1)), t.reshape(-1)) / n
    return ce

# 수치 미분
def dCE_FNN_num(wv, M, K, x, t):
    epsilon = 0.001
    dwv = np.zeros_like(wv)
    for iwv in range(len(wv)):
        wv_modified = wv.copy()
        wv_modified[iwv] = wv[iwv] - epsilon
        mse1 = CE_FNN(wv_modified, M, K, x, t)
        wv_modified[iwv] = wv[iwv] + epsilon
        mse2 = CE_FNN(wv_modified, M, K, x, t)
        dwv[iwv] = (mse2 - mse1) / (2 * epsilon)
    return dwv

# 수치 미분 표시
def show_WV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align='center', color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:], align='center', color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)

# 분류문제 경사하강법 풀이
def Fit_FNN_num(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha):
    wvt = wv_init
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    wv_hist = np.zeros((n ,len(wv_init)))
    epsilon = 0.001
    for i in range(n):
        wvt = wvt - alpha * dCE_FNN_num(wvt, M, K, x_train, t_train)
        err_train[i] = CE_FNN(wvt, M, K, x_train, t_train)
        err_test[i] = CE_FNN(wvt, M, K, x_test, t_test)
        wv_hist[i, :] = wvt
        return wvt, wv_hist, err_train, err_test

# 오차 역전파법 구현
def dCE_FNN(wv, M, K, x, t):
    N, D = x.shape
    w = wv[:M * (D + 1)]
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):]
    v = v.reshape((K, M + 1))
    y, a, z, b = FNN(wv, M, K, x)
    dwv = np.zeros_like(wv)
    dw = np.zeros((M, D + 1))
    dv = np.zeros((K, M + 1))
    delta1 = np.zeors(M)
    delta2 = np.zeors(K)
    for n in range(N):
        for k in range(K):
            delta2[k] = (y[n, k] - t[n, k])
        for j in range(M):
            delta1[j] = z[n, j] * z(1 - z[n, j]) * np.dot(v[:, j], delta2)
        for k in range(K):
            dv[k, :] = dv[k, :] + delta2[k] * z[n, :] / N
        for j in range(M):
            dw[j, :] = dw[j, :] + delta1[j] * np.r_[x[n, :], 1] / N
    dwv = np.c_[dw.reshape((1, M * (D + 1))), dv.reshape((1, K * (M + 1)))]
    dwv = dwv.reshape(-1)
    return dwv

def Show_dWV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align = "center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:], align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)