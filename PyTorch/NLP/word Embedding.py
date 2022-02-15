import torch

# 희소 표현(Sparse Representation) : 대부분 0으로 표현되는 방법
dog = torch.FloatTensor([1, 0, 0, 0, 0])
cat = torch.FloatTensor([0, 1, 0, 0, 0])
computer = torch.FloatTensor([0, 0, 1, 0, 0])
netbook = torch.FloatTensor([0, 0, 0, 1, 0])
book = torch.FloatTensor([0, 0, 0, 0, 1])

# 밀집 표현(Dense Representation) : 단어의 집합 크기로 상정하지 않는 방법
# -> 차원의 크기의 모든 값들이 실수가 됨

# 워드 임베딩(word embedding) : 밀집 벡터의 형태로 표현하는 방법
#   -> 결과 : embedding vector