import pickle
from OOD.utils.Preprocess import Preprocess

f = open("D:\\anaconda3\\envs\\chatbot\\bin\\conversationflowdeteect\\OOD\\train_tools\\dict\\chatbot_dict.bin", "rb")
word_index = pickle.load(f)
f.close()

sent = "여기 부산 타워 어디 있어?"

p = Preprocess(userdic='D:\\anaconda3\\envs\\chatbot\\bin\\conversationflowdeteect\\OOD\\utils\\user_dic.tsv')

pos = p.pos(sent)

keywords = p.get_keywords(pos, without_tag=True)
for word in keywords:
    try:
        print(word, word_index[word])
    except KeyError:
        print(word, word_index['OOV'])