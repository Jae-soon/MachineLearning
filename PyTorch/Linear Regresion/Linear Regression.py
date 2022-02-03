# 훈련 데이터셋 구성
# 예측을 위해 사용하는 데이터를 훈련 데이터셋(training dataset)
# 모델이 얼마나 잘 작동하는지 판별하는 데이터셋을 테스트 데이터셋(test dataset)

# 선형 회귀란 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 일

# 비용 함수(cost function) = 손실 함수(loss function) = 오차 함수(error function) = 목적 함수(objective function)

# 평균제곱오차(MSE) = (실제값 - 예측값) ** 2 / N
#  -> 회귀 문제 해결(MSE가 최소가 되는 가중치와 편향을 구한다)

# 비용 함수의 값을 최소로 하는 가중치와 편향을 찾는 방법 == Optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 가중치 초기화
W = torch.zeros(1, requires_grad=True) 
b = torch.zeros(1, requires_grad=True)



nb_epochs = 1999 # 반복 횟수
for epoch in range(nb_epochs + 1):
    # 가설 (H = Wx + b)
    hypothesis = x_train * W + b

    # 비용 함수 선언
    cost = torch.mean((hypothesis - y_train) ** 2) 

    # 경사 하강법 선언
    optimizer = optim.SGD([W, b], lr=0.01)
    optimizer.zero_grad() # 기울기를 0으로 초기화 -> 미분값이 계속 누적되기 떄문에
    cost.backward() # 비용 함수를 미분하여 기울기 계산
    optimizer.step() # W와 b를 업데이트
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))