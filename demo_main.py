import torch
from transformers import (
    AlbertModel,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    BertTokenizerFast,
)

# 패키지 미설치의 경우 : !pip install transformers로 설치

bert_tokenizer = BertTokenizer.from_pretrained("monologg/kobigbird-bert-base")
model = BertForMaskedLM.from_pretrained("monologg/kobigbird-bert-base")

import torch

a = list(bert_tokenizer.vocab.keys())
context_list = ["김지우", "안양시", "냉장고", "01087124493", "갤럭시s20"]
for ss in context_list:

    text = "나는 경기도 " + ss + "에 살아."
    tokens = bert_tokenizer.tokenize(text)
    token_ids = bert_tokenizer.encode(text)
    result = 0
    result_l = []
    print(token_ids)
    for i, t in enumerate(tokens):
        masked_text_list = []
        masked_text_list = token_ids[:]
        masked_text_list[i + 1] = 4
        # for문 돌 때마다 바뀜
        masked_text = bert_tokenizer.decode(masked_text_list[1:-1])
        print(masked_text)
        encoded = bert_tokenizer(masked_text, return_tensors="pt")
        output = model(**encoded)
        logit = output[0][0]
        logit = torch.softmax(logit, dim=-1)
        index = torch.argmax(logit[i + 1], dim=-1)
        print("best p token : ", bert_tokenizer.decode(index))
        p = float(torch.log(logit[i + 1][token_ids[i + 1]]))
        result_l.append(p)
        result = result + p
        # print(a[token_ids[i+1]])

    print(ss)
    print("sum : ", result * (-1) / len(tokens), "\n\n")
