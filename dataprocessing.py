import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random

max_length = 128

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

file1 = open('DBD/conversation_11.csv', 'r', encoding='utf-8')
file2 = open('DBD/conversation_11_rand.csv', 'r', encoding='utf-8')

lines1 = file1.read().split('\n')
lines2 = file2.read().split('\n')
lines1.pop(0)
lines2.pop(0)

input_ids01 = np.zeros(shape=[len(lines1) * 10, max_length], dtype=np.int32)
input_segments01 = np.zeros(shape=[len(lines1) * 10, max_length], dtype=np.int32)

input_ids02 = np.zeros(shape=[len(lines2) * 10, max_length], dtype=np.int32)
input_segments02 = np.zeros(shape=[len(lines2) * 10, max_length], dtype=np.int32)


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

        length = len(ids)
        if length > max_length:
            length = max_length

        for i in range(length):
            input_ids01[count, i] = ids[i]
            input_segments01[count, i] = segments[i]
        count += 1

print(count, len(lines1))


count = 0

for line2 in lines2:
    TKs = line2.split(',')

    for idx in range(0, len(TKs) - 1):
        tokens = ['[CLS]']
        segments = [0]

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
            input_ids02[count, i] = ids[i]
            input_segments02[count, i] = segments[i]
        count += 1

print(count, len(lines2))

input_ids_ = np.zeros(shape=[count, max_length], dtype=np.int32)
input_segments_ = np.zeros(shape=[count, max_length], dtype=np.int32)

input_ids2_ = np.zeros(shape=[count, max_length], dtype=np.int32)
input_segments2_ = np.zeros(shape=[count, max_length], dtype=np.int32)

for i in range(count):
    input_ids_[i] = input_ids01[i]
    input_ids2_[i] = input_ids02[i]

    input_segments_[i] = input_segments01[i]
    input_segments2_[i] = input_segments02[i]

np.save("input_ids01", input_ids_)
np.save("input_ids02", input_ids2_)
np.save("input_segments01", input_segments_)
np.save("input_segments02", input_segments2_)
