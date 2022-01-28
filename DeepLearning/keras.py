import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from keras.layers.core import Dense, Activation

# 데이터 넣기
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

# 테스트, 훈련 데이터
testRatio = 0.5
x_n_training = int(N * testRatio)
x_train = x[:x_n_training, :]
x_test = x[x_n_training:, :]
t_train = t[:x_n_training, :]
t_test = t[x_n_training:, :]

# 신경망 모델 생성
# Sequential 이라는 유형의 네트워크 model 생성
model = tf.keras.models.Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid', kernel_initializer='uniform'))
model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))
sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

startTime = time.time()
history = model.fit(x_train, t_train, epochs=1000, batch_size=100, verbose=0, validation_data=(x_test, t_test))

# 교차 엔트로피 오차 및 정확도 출력
score = model.evaluate(x_test, t_test, verbose=0)
print("cross entropy {0:3.2f}, accuracy {1:3.2f}".format(score[0], score[1]))
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))