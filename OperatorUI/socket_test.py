import socket

soc = socket.socket()
soc.connect(('localhost', 1234))

def getanswer():
    msg = soc.recv(1024).decode()
    return msg

while True:
    answer = getanswer()
    print(answer)