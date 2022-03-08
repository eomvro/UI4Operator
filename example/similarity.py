from eunjeon import Mecab
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
mecab = Mecab()

def cos_similarity(sen1, sen2):
    sentence = (sen1, sen2)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity

sen1 = "안녕 내 이름은 준영"
sen2 = "안녕하세요 제 이름은 준영입니다"

#JACCARD SIMILARITY
def similarity(sen1, sen2):
    bow1 = mecab.morphs(sen1)
    bow2 = mecab.morphs(sen2)
    cnt = 0
    for token in bow1:
        if token in bow2:
            cnt = cnt + 1
    return cnt / (len(bow1) + len(bow2))

#JACCARD SIMILARITY_변형
def _similarity(sen1, sen2):
    bow1 = mecab.morphs(sen1)
    bow2 = mecab.morphs(sen2)
    bow = mecab.nouns(sen1)

    cnt = 0
    for token in bow1:
        if token in bow2:
            cnt = cnt + 1
    for token2 in bow:
        if token2 in bow2:
            cnt = cnt + 3
    return cnt / (len(bow1) + len(bow2))

query1 = "나는 대학교를 졸업했습니다."
query2 = "나는 고등학교를 졸업했습니다."

print(cos_similarity(query1, query2))
print(similarity(query1, query2))
print(_similarity(query1, query2))
'''
similar1 = []
        for i in range(len(text_casual)):
            bow2 = mecab.morphs(text_casual['Q'][i])
            similar1.append(similarity(bow, bow1, bow2))
        num1 = find_index(max(similar1), similar1)
        number1 = random.choice(num1)
        answer = text_casual['A'][number1]

        probs = model.make_propagate(query, answer)
        # probs[0, 0] = abs(probs[0, 0])
        # probs[0, 1] = abs(probs[0, 1])
        softmaxed_probs = softmax(probs)
'''