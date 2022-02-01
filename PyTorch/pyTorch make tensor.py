import torch

# 1차원 텐서
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t.dim())  # 차원
print(t.shape)  # shape
print(t.size()) # shape

# 2차원 텐서
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])

# Boradcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

"""
tensor([[5., 5.]])
"""

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)

"""
tensor([[4., 5.]])
"""

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

"""
tensor([4., 5.],
       [5., 6.]])
"""

# mean
t = torch.FloatTensor([1, 2])
print(t.mean())

# sum
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거
"""
tensor(10.)
tensor([4., 6.])
tensor([3., 7.])
tensor([3., 7.])
"""

# max, argmax(최댓값 인덱스)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.max())
"""
tensor(4.)
"""

print(t.max(dim=0))
"""
(tensor([3., 4.]), tensor([1, 1])) -> argMax도 함께 출력(dim이 주어졌을 경우)
"""

# view
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
"""
torch.Size([2, 2, 3])
"""

print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경
print(ft.view([-1, 3]).shape)
"""
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
torch.Size([4, 3])
"""

# Squeeze - 1인 차원 제거
ft = torch.FloatTensor([[0], [1], [2]])
print(ft.squeeze())
print(ft.squeeze().shape)
"""
tensor([0., 1., 2.])
torch.Size([3])
"""

# UnSqueeze - 특정 위치에 1인 차원 추가
ft = torch.Tensor([0, 1, 2])
print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(0).shape)
"""
tensor([[0., 1., 2.]])
torch.Size([1, 3])
"""

# Type Casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float())
"""
tensor([1., 2., 3., 4.])
"""
bt = torch.ByteTensor([True, False, False, True])
print(bt)
"""
tensor([1, 0, 0, 1], dtype=torch.uint8)
"""
print(bt.long())
print(bt.float())
"""
tensor([1, 0, 0, 1])
tensor([1., 0., 0., 1.])
"""

# concatenate(연결)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))
"""
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
"""
print(torch.cat([x, y], dim=1))
"""
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
"""

# stacking(쌓음)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
"""
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
"""

# ones_like, zeros_like == np.ones, np.zeros
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.ones_like(x))
"""
tensor([[1., 1., 1.],
        [1., 1., 1.]])
"""

# in-place Operation(덮어쓰기 연산)
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
"""
tensor([[2., 4.],
        [6., 8.]])
"""