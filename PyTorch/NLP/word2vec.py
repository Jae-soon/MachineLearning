# 희소 표현(Sparse Representation) = One-Hot vector
# 단점 : 단어의 유사성을 표현할 수 없음
# 분산 표현(Distributed Represeentation) : 단어의 의미를 다차원 공간에 벡터화
#  - 저차원에 단어의 의미를 여러 차원에다가 분산하여 표현
#  - 이런 표현 방법을 사용하면 단어 간 유사도를 계산

# CBOW(Continuous Bag of Words)
#   - 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법
