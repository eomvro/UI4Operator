from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, join_room, leave_room, emit
from flask_session import Session
import pandas as pd
import threading

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'secret'
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)
socketio = SocketIO(app, manage_session=False)

df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv',encoding='euc-kr', index_col=0)

#app
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if(request.method=='POST'):
        username = request.form['username']
        room = request.form['room']
        session['username'] = username
        session['room'] = room
        return render_template('chat.html', session=session)
    else:
        if(session.get('username') is not None):
            return render_template('chat.html', session=session)
        else:
            return redirect(url_for('index'))

@socketio.on('join', namespace='/chat')
def join(message):
    df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv', encoding='euc-kr', index_col = 0)
    room = session.get('room')
    join_room(room)
    #emit('status', {'msg' : session.get('username') + ' has entered the room'}, room=room)
    len = df_conversation.__len__()
    for i in range(len):
        if (df_conversation['answer'][i] == '-'):
            emit('message', {'msg': 'User : ' + df_conversation['question'][i]}, room=room)
        else:
            emit('message', {'msg': 'User : ' + df_conversation['question'][i]}, room=room)
            emit('message', {'msg': 'Me   : ' + df_conversation['answer'][i]}, room=room)
    if (df_conversation['answer'][len-1] == '-'):
        emit('message', {'msg': '***ANSWER THE USER***'}, room=room)

    def update_csv():
        df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv', encoding='euc-kr', index_col=0)
        length = df_conversation.__len__()
        threading.Timer(5, update_csv).start()
    update_csv()

@socketio.on('text', namespace='/chat')
def text(message):
    df_conversation = pd.read_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv', encoding='euc-kr', index_col = 0)
    room = session.get('room')
    #emit('message', {'msg' : session.get('username') + ' : ' + message['msg']}, room=room)
    len = df_conversation.__len__()
    emit('message', {'msg': 'Me   : ' + message['msg']}, room=room)
    df_conversation['answer'][len-1] = message['msg']
    df2 = pd.DataFrame([['-', '', '']], columns=['question', 'answer', 'breakdown'])
    df_conversation = df_conversation.append(df2, ignore_index=True)
    df_conversation.to_csv(f'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\actual_conversation\\user.csv', encoding='euc-kr')
    #print(message['msg'])

@socketio.on('left', namespace='/chat')
def left(message):
    room = session.get('room')
    username = session.get('username')
    leave_room(room)
    session.clear()
    #emit('status', {'msg' : username + ' has left the room'}, room=room)

if __name__=="__main__":
    socketio.run(app)