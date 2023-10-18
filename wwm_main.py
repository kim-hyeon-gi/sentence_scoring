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
context_list = ["김지우", "안양시", "냉장고", "01087124493", "갤럭시s20"]
for ss in context_list:
    text = "나는 경기도 " + ss + "에 살아."
    token_ids = bert_tokenizer.encode(text)
    text_list = text.split(" ")
    result = 0
    result_l = []
    masked_start = 1  # cls
    for i, t in enumerate(text_list):
        masked_text_list = text_list[:]
        masked_token_len = len(bert_tokenizer.tokenize(masked_text_list[i]))
        masked_text_list[i] = "[MASK]" * masked_token_len
        masked_text = " ".join(masked_text_list)
        print(masked_text)

        encoded = bert_tokenizer(masked_text, return_tensors="pt")
        output = model(**encoded)
        logit = output[0][0]
        # logit = torch.softmax(logit,dim = -1)

        for j in range(masked_start, masked_start + masked_token_len):
            p = float(logit[j][token_ids[j]])
            result_l.append(p)
            result = result + p
            index = torch.argmax(logit[j], dim=-1)
            print("best p token : ", bert_tokenizer.decode(index))
        masked_start = masked_start + masked_token_len
    print(result)
    print(result_l)
    print("sum : ", result / (len(token_ids) - 2), "\n\n")
