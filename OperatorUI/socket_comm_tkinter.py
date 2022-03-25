import socket
from tkinter import *
import threading
from tkinter import ttk

from click import style
from matplotlib import image

# soc = socket.socket()
# soc.connect(('localhost', 1234))

BG_GRAY = "thistle"
BG_COLOR = "lavenderblush"
EN_COLOR = "whitesmoke"
TEXT_COLOR = "black"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class ChatApplication(Tk):
    ip = "192.168.50.173"
    port = 1234

    def __init__(self):
        Tk.__init__(self)
        self.conn_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn_soc.connect((ChatApplication.ip, ChatApplication.port))
        self.curFrame = None
        self._frameA = Avatar("avatarA", self)
        self._frameB = Avatar("avatarB", self)
        self.title("Chat")
        self.geometry("470x600+100+100")
        self.resizable(False, False)

        # head label
        head_label = Label(
            self,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="용두산 공원 챗봇",
            font=FONT_BOLD,
            pady=10,
            width=470,
        )
        # head_label.place(relwidth=1)
        head_label.pack(side="top")
        selectAvatar = Label(self, bg="gray", fg=TEXT_COLOR, pady=10, width=470)
        selectAvatar.pack(side="top")

        # select button
        self.avatarA = Button(
            selectAvatar,
            text="avatarA",
            command=lambda: self.switch_frame(self._frameA),
        )
        self.avatarB = Button(
            selectAvatar,
            text="avatarB",
            command=lambda: self.switch_frame(self._frameB),
        )

        self.avatarA.place(relx=0, rely=0.1, relheight=1, relwidth=0.22)
        self.avatarB.place(relx=0.2, rely=0.1, relheight=1, relwidth=0.22)

        self.switch_frame(self._frameA)

        th = threading.Thread(target=self.update_conversation)
        th.start()

    def switch_frame(self, selected_frame):
        if self.curFrame is not None:
            self.curFrame.place_forget()
        self.curFrame = selected_frame
        self.curFrame.place(relx=0, rely=0.13, relheight=0.87, relwidth=1)
        if self.curFrame == self._frameA:
            self.avatarA.configure(bg=BG_COLOR)
            self.avatarB.configure(bg=BG_GRAY)
        else:
            self.avatarA.configure(bg=BG_GRAY)
            self.avatarB.configure(bg=BG_COLOR)

    def alarm(self):
        if self.curFrame == self._frameA:
            self.avatarA.configure(bg="red")
        else:
            self.avatarB.configure(bg="red")

    def update_conversation(self):
        while True:
            msg = self.conn_soc.recv(1024).decode()
            if not msg:
                return
            self.avatarA.configure(bg="red")
            # self._frameA.msg_entry.delete(0, END)
            msg1 = f"{msg}\n"
            self._frameA.text_widget.configure(state=NORMAL)
            self._frameA.text_widget.insert(END, msg1)
            self._frameA.text_widget.configure(state=DISABLED)
            self._frameA.text_widget.see(END)

    # def __init__(self):
    #     self.conn_soc = None
    #     self.window = None
    #     self.text_widget = None
    #     self.msg_entry = None

    # def run(self):
    #     self.conn_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     self.conn_soc.connect((ChatApplication.ip, ChatApplication.port))
    #     self.window = Tk()
    #     self._setup_main_window()

    #     self.window.mainloop()

    # def _setup_main_window(self):
    #     self.window.title("Chat")
    #     self.window.resizable(width=False, height=False)
    #     self.window.configure(width=470, height=550, bg=BG_COLOR)

    #     # head label
    #     head_label = Label(
    #         self.window,
    #         bg=BG_COLOR,
    #         fg=TEXT_COLOR,
    #         text="용두산 공원 챗봇",
    #         font=FONT_BOLD,
    #         pady=10,
    #     )
    #     head_label.place(relwidth=1)

    #     # tiny divider
    #     line = Label(self.window, width=450, bg=BG_GRAY)
    #     line.place(relwidth=1, rely=0.07, relheight=0.012)

    #     # text widget
    #     self.text_widget = Text(
    #         self.window,
    #         width=20,
    #         height=2,
    #         bg=BG_COLOR,
    #         fg=TEXT_COLOR,
    #         font=FONT,
    #         padx=5,
    #         pady=5,
    #     )
    #     self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
    #     self.text_widget.configure(cursor="arrow", state=DISABLED)

    #     # bottom label
    #     bottom_label = Label(self.window, bg=BG_GRAY, height=80)
    #     bottom_label.place(relwidth=1, rely=0.825)

    #     # message entry box
    #     self.msg_entry = Entry(bottom_label, bg=EN_COLOR, fg=TEXT_COLOR, font=FONT)
    #     self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
    #     self.msg_entry.focus()
    #     self.msg_entry.bind("<Return>", self._on_enter_pressed)

    #     # send button
    #     send_button = Button(
    #         bottom_label,
    #         text="send",
    #         font="FONT_BOLD",
    #         width=20,
    #         bg=BG_GRAY,
    #         command=lambda: self._on_enter_pressed(None),
    #     )
    #     send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    # def _on_enter_pressed(self, event):
    #     msg = self.msg_entry.get()
    #     self._insert_message(msg)

    # def _insert_message(self, msg):
    #     if not msg:
    #         return

    #     self.msg_entry.delete(0, END)

    #     msg1 = f"You : {msg}\n"
    #     self.text_widget.configure(state=NORMAL)
    #     self.text_widget.insert(END, msg1)
    #     self.text_widget.configure(state=DISABLED)

    #     self.text_widget.see(END)
    #     self.conn_soc.send(bytes(msg, "utf-8"))


class Avatar(Frame):
    def __init__(self, name, app):
        self.app = app
        super(Avatar, self).__init__()
        self.name = name
        self.text_widget = Text(
            self,
            width=20,
            height=2,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=FONT,
            padx=5,
            pady=5,
        )
        self.text_widget.place(relheight=0.825, relwidth=1)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # bottom label
        bottom_label = Label(self, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg=EN_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = Button(
            bottom_label,
            text="send",
            font="FONT_BOLD",
            width=20,
            bg=BG_GRAY,
            command=lambda: self._on_enter_pressed("<Retrun>"),
        )
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _insert_message(self, msg):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        if msg == "new":
            self.app.alarm()

        msg1 = f"You : {msg}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)
        app.conn_soc.send(bytes(msg, "utf-8"))

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg)


# def gettext():
#     msg = soc.recv(1024).decode()
#     return msg

if __name__ == "__main__":
    app = ChatApplication()
    app.mainloop()

###0207 오후 4시 반 현재 상황
###첫 대화는 오는데 그 후 대화 업데이트가 안됨
