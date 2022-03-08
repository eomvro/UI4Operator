import pandas as pd

#friend_dict_list = [{'question' : '안녕하세요', 'answer' : '안녕 만나서 반가워', 'breakdown' : ''},
#                    {'question' : '뭐해?', 'answer' : '일하고 있죠.', 'breakdown' : ''}]

num = input('user 번호 입력 : ')

actual_conversation_user = []
df_conversation = pd.DataFrame(actual_conversation_user, columns = ['question', 'answer', 'breakdown'])

df_conversation.to_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\conversation_user{num}.csv', encoding='euc-kr')
df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\conversation_user{num}.csv', encoding='euc-kr')

for i in range(5):
    question = input('question : ')
    answer = input('answer : ')

    if answer.__contains__('-1'):
        breakdown = 'V'
        answer = answer.strip('-1')
    else:
        breakdown = ''

    df2 = pd.DataFrame([
        [question, answer, breakdown]
    ], columns=['question', 'answer', 'breakdown'])

    df_conversation = df_conversation.append(df2, ignore_index=True)
    df_conversation.to_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\conversation_user{num}.csv', encoding='euc-kr')