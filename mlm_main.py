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

a = list(bert_tokenizer.vocab.keys())
context_list = ["김지우", "안양시", "냉장고", "010 8712 4493", "갤럭시s20"]
for ss in context_list:

    text = "내 전화번호는 " + ss + "니까 연락해."
    tokens = bert_tokenizer.tokenize(text)
    token_ids = bert_tokenizer.encode(text)
    result = 0
    result_l = []
    for i, t in enumerate(tokens):
        encoded = bert_tokenizer(text, return_tensors="pt")
        encoded["input_ids"][0][i + 1] = 4
        print("decode : ", bert_tokenizer.decode(encoded["input_ids"][0]))
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
