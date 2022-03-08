from OOD.utils.Preprocess import Preprocess

sent = '이 근처 맛집 추천해줘'

p = Preprocess(userdic='D:\\anaconda3\\envs\\chatbot\\bin\\conversationflowdeteect\\OOD\\utils\\user_dic.tsv')
pos = p.pos(sent)

ret = p.get_keywords(pos, without_tag=False)
print(ret)

ret = p.get_keywords(pos, without_tag=True)
print(ret)