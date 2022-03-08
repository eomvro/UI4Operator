# 소켓을 사용하기 위해서는 socket을 import해야 한다.
import socket, threading
from makeanswer2 import makeanswer
from eunjeon import Mecab
import pandas as pd
mecab = Mecab()
import speech_recognition as sr
r = sr.Recognizer()
import time

#ser = socket.socket()
#ser.bind(("localhost", 1234))
#ser.listen(3)

def decode_str(sentence):
    result = ''
    tks = str(sentence).split('##')
    for tk in tks:
        try:
            idx = int(tk)
            result += str(chr(idx))
        except:
            result += tk
    return result

def make_finalanswer():
    df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv',encoding='euc-kr', index_col=0)
    len = df_conversation.__len__()
    answer = df_conversation['answer'][len-1]
    if answer != "-":
        return answer
    threading.Timer(1, make_finalanswer).start()

#user_num = input('user 번호 입력 : ')

print('대기중')
# binder함수는 서버에서 accept가 되면 생성되는 socket 인스턴스를 통해 client로 부터 데이터를 받으면 echo형태로 재송신하는 메소드이다.

#actual_conversation_user = []
#df_conversation = pd.DataFrame(actual_conversation_user, columns = ['question', 'answer', 'breakdown'])
#df_conversation.to_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user_{user_num}.csv', encoding='euc-kr')

def binder(client_socket, addr):
    # 커넥션이 되면 접속 주소가 나온다.
    #print('Connected by', addr)
    try:
        # 접속 상태에서는 클라이언트로 부터 받을 데이터를 무한 대기한다.
        # 만약 접속이 끊기게 된다면 except가 발생해서 접속이 끊기게 된다.
        while True:
            #c, adress = ser.accept()
            # socket의 recv함수는 연결된 소켓으로부터 데이터를 받을 대기하는 함수입니다. 최초 4바이트를 대기합니다.
            data = client_socket.recv(4)
            # 최초 4바이트는 전송할 데이터의 크기이다. 그 크기는 big 엔디언으로 byte에서 int형식으로 변환한다.
            # C#의 BitConverter는 big엔디언으로 처리된다.
            length = int.from_bytes(data, "big")
            # 다시 데이터를 수신한다.
            data = client_socket.recv(length)
            # 수신된 데이터를 str형식으로 decode한다.
            msg = data.decode()
            # 수신된 메시지를 콘솔에 출력한다.
            decoded_msg = decode_str(msg)
            question = decoded_msg[1:]
            print(question)

            #####
            #c.send(bytes('User    : ' + question, 'utf-8'))

            bow1 = mecab.morphs(question)
            bow = mecab.nouns(question)

            text = ''
            for i in range(len(bow)):
                text = text + ' ' + bow[i]
            print('User  : ' + question + '   ' + '\033[95m' + text + '\033[0m')
            #  여기서 casual과 knowledge 구분하는 함수나 스크립트 추가해서 if casual 이면 makeanswer_casual / knowledge면 makeanswer_knowledge

            answer = makeanswer(question)
            print(answer)

            #####
            #c.send(bytes('Chatbot : ' + answer, 'utf-8'))

            df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv', encoding='euc-kr', index_col = 0)

            if answer.__contains__('-1'):
                print('\033[95m' + "!!OPERATOR 연결!!" + '\033[0m')
                breakdown = 'V'
                df2 = pd.DataFrame([[question, '-', breakdown]], columns=['question', 'answer', 'breakdown'])
                df_conversation = df_conversation.append(df2, ignore_index=True)
                df_conversation.to_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv',encoding='euc-kr')
                #len = df_conversation.__len__()
                #여기서 for while 이런걸로 df_conversation[answer][len-1] != '-' 될때까지 기다린 담에 되면 final answer 로 내보내면 되지 않을까?
                #max_time_end = time.time() + (10)  # 10초
                #while df_conversation['answer'][len-1] == '-':
                #    print(time.time())
                #    if time.time() > max_time_end:
                #        break
                answer = "대화가 원활하지 않아 상담원을 연결합니다"

            else:
                breakdown = ''
                df2 = pd.DataFrame([[question, answer, breakdown]], columns=['question', 'answer', 'breakdown'])
                df_conversation = df_conversation.append(df2, ignore_index=True)
                df_conversation.to_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv',encoding='euc-kr')

            print('Agent : ', answer)

            # 바이너리(byte)형식으로 변환한다.
            data = answer.encode()
            # 바이너리의 데이터 사이즈를 구한다.
            length = len(data)
            # 데이터를 클라이언트로 전송한다.
            client_socket.sendall(data)
            client_socket.sendall(length.to_bytes(4, byteorder='little'))
            # 데이터 사이즈를 big 엔디언 형식으로 byte로 변환한 다음 전송한다.(※이게 버그인지 big을 써도 little엔디언으로 전송된다.)

    except:
        print("except : ", addr)

    finally:
        # 접속이 끊기면 socket 리소스를 닫는다.
        client_socket.close()

# 소켓을 만든다.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 소켓 레벨과 데이터 형태를 설정한다.
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# 서버는 복수 ip를 사용하는 pc의 경우는 ip를 지정하고 그렇지 않으면 None이 아닌 ''로 설정한다.
# 포트는 pc내에서 비어있는 포트를 사용한다. cmd에서 netstat -an | find "LISTEN"으로 확인할 수 있다.
server_socket.bind(('', 8888))
# server 설정이 완료되면 listen를 시작한다.
server_socket.listen()

try:
    # 서버는 여러 클라이언트를 상대하기 때문에 무한 루프를 사용한다.
    while True:
        # client로 접속이 발생하면 accept가 발생한다.
        # 그럼 client 소켓과 addr(주소)를 튜플로 받는다.
        client_socket, addr = server_socket.accept()
        th = threading.Thread(target=binder, args=(client_socket, addr))
        # 쓰레드를 이용해서 client 접속 대기를 만들고 다시 accept로 넘어가서 다른 client를 대기한다.
        th.start()
except:
    print("server")
finally:
    # 에러가 발생하면 서버 소켓을 닫는다.
    server_socket.close()