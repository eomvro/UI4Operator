import json
import numpy as np

import tokenization
from transformers import AutoTokenizer, AutoModelForMaskedLM


label_dictionary = {
    "org:founded": "설립",
    "org:place_of_headquarters": "본사",
    "org:alternate_names": "대체 이름",
    "org:member_of": "멤버",
    "org:members": "멤버",
    "org:product": "제품",
    "org:founded_by": "설립자",
    "org:top_members/employees": "고용인",
    "org:number_of_employees/members": "고용 숫자",
    "per:date_of_birth": "출생일",
    "per:date_of_death": "사망일",
    "per:place_of_birth": "출생지",
    "per:place_of_death": "사망지",
    "per:place_of_residence": "거주지",
    "per:origin": "본원",
    "per:employee_of": "근무지",
    "per:schools_attended": "학력",
    "per:alternate_names": "별명",
    "per:parents": "부모",
    "per:children": "자식",
    "per:siblings": "형제",
    "per:spouse": "배우자",
    "per:other_family": "가족",
    "per:colleagues": "동료",
    "per:product": "제품",
    "per:religion": "종교",
    "per:title": "제목"
}

##vocab = tokenization.load_vocab('vocab.txt')
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

file = open('wos-v1_train.json', 'r', encoding='utf-8')
file2 = open('wos_sentence_files', 'w', encoding='utf-8')

objects = json.load(file)

max_length = 64
batch_size = len(objects)

num_sentences = 0
whole_sentences = []

for i in range(len(objects)):
    dialogues = objects[i]['dialogue']

    for dialogue in dialogues:
        sentence = dialogue['text']

        file2.write(sentence + '\n')
file2.close()
file.close()