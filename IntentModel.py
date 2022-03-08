from OOD.utils.Preprocess import Preprocess
from OOD.model.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='D:\\anaconda3\\envs\\chatbot\\bin\\conversationflowdeteect\\OOD\\train_tools\\dict\\chatbot_dict.bin',
               userdic='D:\\anaconda3\\envs\\chatbot\\bin\\conversationflowdeteect\\OOD\\utils\\user_dic.tsv')

intent = IntentModel(model_name='D:\\anaconda3\\envs\\chatbot\\bin\\conversationflowdeteect\\OOD\\model\\intent\\intent_model.h5', proprocess=p)

def IntentModel(query):
    predict = intent.predict_class(query)
    predict_label = intent.labels[predict]
    return predict, predict_label