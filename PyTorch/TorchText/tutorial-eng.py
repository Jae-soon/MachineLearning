from torchtext.legacy.data import TabularDataset
from torchtext import data
# 훈련 데이터와 테스트 데이터 분류
import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# 필드 생성
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

# 단어집합 생성 (min_freq == 단어집합 추가 시 최소 등장 빈도 설정)
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
# 집합 내 단어 확인
print(TEXT.vocab.stoi)

# 토치텍스트 데이터로더 만들기
from torchtext.data import Iterator

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)
# 출력 = 5000(25000개의 데이터를 5개의 배치로 묶었기 때문에)

batch = next(iter(train_loader))
