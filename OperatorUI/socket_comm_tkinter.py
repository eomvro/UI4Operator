import socket
from tkinter import *
import threading

# soc = socket.socket()
# soc.connect(('localhost', 1234))

BG_GRAY = "thistle"
BG_COLOR = "lavenderblush"
EN_COLOR = "whitesmoke"
TEXT_COLOR = 'black'

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    ip = 'localhost'
    port = 1234

    def __init__(self):
        self.conn_soc = None
        self.window = None
        self.text_widget=None
        self.msg_entry=None

    def run(self):
        self.conn_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn_soc.connect((ChatApplication.ip, ChatApplication.port))
        self.window = Tk()
        self._setup_main_window()
        th = threading.Thread(target=self.update_conversation)
        th.start()
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="용두산 공원 챗봇", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg=EN_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = Button(bottom_label, text="send", font="FONT_BOLD", width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg)

    def _insert_message(self, msg):
        if not msg:
            return

        self.msg_entry.delete(0, END)

        msg1 = f"You : {msg}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)
        self.conn_soc.send(bytes(msg, 'utf-8'))

    def update_conversation(self):
        while True:
            msg = self.conn_soc.recv(1024).decode()
            if not msg:
                return

            self.msg_entry.delete(0, END)
            msg1 = f"{msg}\n\n"
            self.text_widget.configure(state=NORMAL)
            self.text_widget.insert(END, msg1)
            self.text_widget.configure(state=DISABLED)
            self.text_widget.see(END)

# def gettext():
#     msg = soc.recv(1024).decode()
#     return msg

if __name__ == "__main__":
    app = ChatApplication()
    app.run()

###0207 오후 4시 반 현재 상황
###첫 대화는 오는데 그 후 대화 업데이트가 안됨