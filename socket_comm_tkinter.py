import socket, threading
from makeanswer2 import makeanswer
from eunjeon import Mecab
mecab = Mecab()

ser = socket.socket()
ser.bind(("localhost", 1234))
ser.listen(3)

c, adress = ser.accept()

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

print('대기중')

def binder(client_socket, addr):
    try:
        while True:
            print("connected_operatorUI")
            # data = client_socket.recv(4)
            # length = int.from_bytes(data, "big")
            # data = client_socket.recv(length)
            # msg = data.decode()
            # decoded_msg = decode_str(msg)
            # question = decoded_msg[1:]
            #
            # answer = makeanswer(question)


            '''
                        if answer.__contains__('-1'):
                answer = '***대답해주세요***'
                msg = '***대답해주세요***\n' + 'User     : ' + question
                #msg = answer

            else:
                answer = answer
                msg = 'User     : ' + question + '\n\n' + 'Chatbot : ' + answer + '\n\n'
                #msg = answer
            '''

            # msg = 'User     : ' + question + '\n\n' + 'Chatbot : ' + answer + '\n\n'
            #
            # print('client : ' + c.recv(1024).decode())
            msg = input("you : ")
            c.send(bytes(msg, 'utf-8'))
            # data = answer.encode()
            # length = len(data)
            # client_socket.sendall(data)
            # client_socket.sendall(length.to_bytes(4, byteorder='little'))

            #####
            c.send(bytes(msg, 'utf-8'))

    except:
        print("except : ", addr)

    finally:
        client_socket.close()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('', 8888))
server_socket.listen()

try:
    while True:
        # client_socket, addr = ser.accept()
        # th = threading.Thread(target=binder, args=(client_socket, addr))
        # th.start()
        # print('client : ' + c.recv(1024).decode())
        msg = input("you : ")
        c.send(bytes('avatar: '+msg, 'utf-8'))
except:
    print("server")

finally:
    server_socket.close()

##0204 15시 현재 상황 대화 계속 이어질 수 있고 OperatorUI(tkinter program)으로 계속해서 넘어감(콘솔 창에 업데이트 ㅇ)
##해야할 사항 : 1. 넘어간 대화들 채팅창에서 업데이트
            ##2. 오퍼레이터가 필요시 채팅한 내용을 입력받아 넘어올 수 있도록