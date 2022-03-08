import Model

#계속 firstTraining = true
#학습 전에 하던거랑 이어서 할때는 is continue true
model = Model.KoNET(firstTraining=True, testCase=True)
#model.Training(is_Continue=False, training_epoch=61913)
model.Test(is_Continue=False,  training_epoch=4000)
#model.propagate(is_Continue=False, training_epoch=20000)

model.model_setting()

while True:
    sentence1 = input('sentence1')
    sentence2 = input('sentence2')

    print(model.make_propagate(sentence1, sentence2))