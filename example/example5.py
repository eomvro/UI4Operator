import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

max_length = 128

file1 = open('../DBD/korean_conversation_21.csv', 'r', encoding='utf-8')
lines1 = file1.read().split('\n')
lines1.pop(0)

input_ids_ex = np.zeros(shape=[len(lines1) * 10, max_length], dtype=np.int32)
input_segments_ex = np.zeros(shape=[len(lines1) * 10, max_length], dtype=np.int32)

count = 0

for line1 in lines1:
    TKs = line1.split(',')

    for idx in range(0, len(TKs) - 1):
        tokens = ['[CLS]']
        segments = [0]
        #TKs는 conversation dataset에서 원소? 하나하나
        #idx :0~4(5개)
        #다 tokenize함
        tokens_ = tokenizer.tokenize(TKs[idx])

        for token in tokens_:
            tokens.append(token)
            segments.append(0)
        tokens.append('[SEP]')
        segments.append(0)

        tokens_ = tokenizer.tokenize(TKs[idx + 1])

        for token in tokens_:
            tokens.append(token)
            segments.append(1)
        tokens.append('[SEP]')
        segments.append(1)

        ids = tokenizer.convert_tokens_to_ids(tokens=tokens)

        #print(ids)
        #print(segments)

        length = len(ids)
        if length > max_length:
            length = max_length

        for i in range(length):
            input_ids_ex[count, i] = ids[i]
            input_segments_ex[count, i] = segments[i]
        count += 1

input_ids_11 = np.zeros(shape=[count, max_length], dtype=np.int32)
input_segments_11 = np.zeros(shape=[count, max_length], dtype=np.int32)

for i in range(count):
    input_ids_11[i] = input_ids_ex[i]
    input_segments_11[i] = input_segments_ex[i]

np.save("input_ids_ex", input_ids_11)
np.save("input_segments_ex", input_segments_11)

input_ids = np.load('../DBD/input_ids_ex.npy')
r_ix_ex = np.array(range(4000, input_ids.shape[0]), dtype=np.int32)