import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')

# 필드 정의
from torchtext import data # torchtext.data 임포트
from konlpy.tag import Mecab

# 형태소 분석기 사용
tokenizer = Mecab()
ID = data.Field(sequential = False,
                use_vocab = False) # 실제 사용은 하지 않을 예정

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs, # 토크나이저로는 Mecab 사용.
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True)

# 데이터셋 만들기
from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
        path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

# 단어 집합 생성
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

# 데이터로더 생성
from torchtext.data import Iterator

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)
batch = next(iter(train_loader)) # 첫번째 미니배치