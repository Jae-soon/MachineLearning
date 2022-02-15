from konlpy.tag import Okt  
okt = Okt()  
token = okt.morphs("나는 자연어 처리를 배운다")  

# 고유 인덱스 지정
word2index = {}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca] = len(word2index)

# 원-핫 벡터 생성 함수
def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index = word2index[word]
       one_hot_vector[index] = 1
       return one_hot_vector

one_hot_encoding("자연어",word2index)
"""
출력 : [0, 0, 1, 0, 0, 0]  
"""

