import numpy as np
import matplotlib.pyplot as plt
from . import logistic
from scipy.optimize import minimize

np.random.seed(seed=0)
x_min = 0
x_max = 2.5
x_n = 30
x_col = ['cornflowerblue', 'gray']
x = np.zeros(x_n)
t = np.zeros(x_n, dtype=np.uint8)
dist_s = [0.4, 0.8] # 분포의 시작지점
dist_w = [0.8, 1.6] # 분포의 폭
pi = 0.5 # 클래스 0의 비율(1/2)
for n in range(x_n):
    wk = np.random.rand()
    t[n] = 0 * (wk < pi) + 1 * (wk >= pi)
    x[n] = np.random.rand() * dist_w[t[n]] + dist_s[t[n]]
    
print('X=' + str(np.round(x, 2)))
print('T=' + str(t))

# 데이터 보여주기
def show_data1(x, t):
    k = np.max(t) + 1
    for k in range(k):
        plt.plot(x[t == k], t[t == k], x_col[k], alpha=0.5, linestyle='none', marker='o')
        plt.grid(True)
        plt.ylim(-.5, 1.5)
        plt.xlim(x_min, x_max)
        plt.yticks([0, 1])
        

# logistic 회귀모델 출력 -> sigmoid
def show_logistic(w):
    xb = np.linspace(x_min, x_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    
    # 결정경계
    i = np.min(np.where(y > 0.5))
    b = (xb[i - 1] + xb[i]) / 2
    plt.plot([b, b], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    return b

#평균 교차 엔트로피 오차
def cee_logistic(w, x, t):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee = cee / x_n
    return cee

# 경사 하강법을 위한 평균 교차 엔트로피 오차의 미분
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / x_n
    return dcee

# 경사 하강법
def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistic, w_init, args=(x, t), jac=dcee_logistic, method="CG")
    return res1.x