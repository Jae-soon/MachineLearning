import numpy as np

# Logistic 회계모델 -> (= sigmoid)
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y