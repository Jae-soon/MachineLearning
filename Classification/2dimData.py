import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

np.random.seed(seed=1)
N = 100
K = 3
t3 = np.zeros((N, 3), dtype=np.uint8)
t2 = np.zeros((N, 2), dtype=np.uint8)
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
            t3[n, k] = 1
            break
    for k in range(2):
        x[n, k] = (np.random.randn() * sig[t3[n, :] == 1, k] + mu[t3[n, :] == 1, k])
t2[:, 0] = t3[:, 0]
t2[:, 1] = t3[:, 1] | t3[:, 2]

# 데이터 분포도 보기
def show_data2(x, t):
    wk, K = t.shape
    c = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1], linestyle='none', markeredgecolor='black', marker='o', color=c[k], alpha=0.8)
        plt.grid(True)

# 로지스틱 회귀 모델
def logistic2(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y

# 모델 3D 보기
def show3d_logistic2(ax, w):
    xn = 50
    x0 = np.linspace(x_range0[0], x_range0[1], xn)
    x1 = np.linspace(x_range1[0], x_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray', rstride=5, cstride=5, alpha=0.3)
    
def show_data2_3d(ax, x, t):
    c = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i, marker='o', color=c[i], markeredgecolor='black', linestyle='none', markersize=5, alpha=0.8)
    ax.view_init(elev=25, azim=-30)

# 크로스 엔트로피 오차
def cee_logistic2(w, x, t):
    x_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) + (1 - t[n, 0]) * np.log(1-y[n]))
    cee = cee / x_n
    return cee

# 크로스 엔트로피 오차의 미분
def dcee_logistic2(w, x, t):
    x_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / x_n
    return dcee