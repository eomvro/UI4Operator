import imp
from tkinter import *

from tkinter import ttk

from click import style
from matplotlib import image



        
        

BG_GRAY = "thistle"
LEFT_COLOR = "blue"
BG_COLOR = "lavenderblush"
EN_COLOR = "whitesmoke"
TEXT_COLOR = 'black'

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class SampleApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.curFrame = None
        self._frameA = Avatar("avatarA", self)
        self._frameB = Avatar("avatarB", self)
        self.title("Chat")
        self.geometry("470x600+100+100")
        self.resizable(False, False)

        # head label
        head_label = Label(self, bg=BG_COLOR, fg=TEXT_COLOR,
                            text="용두산 공원 챗봇", font=FONT_BOLD, pady=10, width= 470)
        # head_label.place(relwidth=1)
        head_label.pack(side='top')
        selectAvatar = Label(self, bg="gray", fg=TEXT_COLOR, pady=10, width= 470)
        selectAvatar.pack(side='top')

        # select button
        self.avatarA = Button(selectAvatar, text="avatarA", command=lambda:self.switch_frame(self._frameA))
        self.avatarB = Button(selectAvatar, text="avatarB", command=lambda:self.switch_frame(self._frameB))

        self.avatarA.place(relx=0, rely=0.1, relheight=1, relwidth=0.22)
        self.avatarB.place(relx=0.2, rely=0.1, relheight=1, relwidth=0.22)
        
        self.switch_frame(self._frameA)
        

    def switch_frame(self, selected_frame):
        if self.curFrame is not None:
            self.curFrame.place_forget()
        self.curFrame = selected_frame
        self.curFrame.place(relx=0, rely=0.13, relheight=0.87, relwidth=1)
        if(self.curFrame == self._frameA):
          self.avatarA.configure(bg = BG_COLOR)
          self.avatarB.configure(bg = BG_GRAY)
        else:
          self.avatarA.configure(bg = BG_GRAY)
          self.avatarB.configure(bg = BG_COLOR)
    def alarm(self):
      if(self.curFrame == self._frameA):
        self.avatarA.configure(bg = "red")
      else:
        self.avatarB.configure(bg = "red")

class Avatar(Frame):
  def __init__(self, name, app):
    self.app = app
    super(Avatar, self).__init__()
    self.name = name
    self.text_widget = Text(self, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                        font=FONT, padx=5, pady=5)
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
    send_button = Button(bottom_label, text="send", font="FONT_BOLD", width=20, bg=BG_GRAY,
                          command=lambda: self._on_enter_pressed("<Retrun>"))
    send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
  def _insert_message(self,msg):
    if not msg:
        return

    self.msg_entry.delete(0, END)
    if msg == 'new':
      self.app.alarm()

    msg1 = f"You : {msg}\n"
    self.text_widget.configure(state=NORMAL)
    self.text_widget.insert(END, msg1)
    self.text_widget.configure(state=DISABLED)

    self.text_widget.see(END)
    
  def _on_enter_pressed(self,event):
        msg = self.msg_entry.get()
        self._insert_message(msg)
        
  

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()



# ################## frame1 ######################
# frame1=Frame(root)

# # text widget
# frame1.text_widget = Text(frame1, width=20, height=2, bg="blue", fg=TEXT_COLOR,
#                         font=FONT, padx=5, pady=5)
# frame1.text_widget.place(relheight=0.825, relwidth=1)
# frame1.text_widget.configure(cursor="arrow", state=DISABLED)

# # bottom label
# bottom_label1 = Label(frame1, bg=BG_GRAY, height=80)
# bottom_label1.place(relwidth=1, rely=0.825)

# # message entry box
# frame1.msg_entry = Entry(bottom_label1, bg=EN_COLOR, fg=TEXT_COLOR, font=FONT)
# frame1.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
# frame1.msg_entry.focus()
# frame1.msg_entry.bind("<Return>", _on_enter_pressed)

# # send button
# send_button1 = Button(bottom_label1, text="send", font="FONT_BOLD", width=20, bg=BG_GRAY,
#                       command=lambda: _on_enter_pressed(frame1))
# send_button1.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

# frame1.place(relx=0, rely=0.13, relheight=0.87, relwidth=1)


# ################## frame2 ######################
# frame2=Frame(root)

# # text widget
# frame2.text_widget = Text(frame2, width=20, height=2, bg="green", fg=TEXT_COLOR,
#                         font=FONT, padx=5, pady=5)
# frame2.text_widget.place(relheight=0.825, relwidth=1)
# frame2.text_widget.configure(cursor="arrow", state=DISABLED)

# # bottom label
# bottom_label2 = Label(frame2, bg=BG_GRAY, height=80)
# bottom_label2.place(relwidth=1, rely=0.825)

# # message entry box
# frame2.msg_entry = Entry(bottom_label2, bg=EN_COLOR, fg=TEXT_COLOR, font=FONT)
# frame2.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
# frame2.msg_entry.focus()
# frame2.msg_entry.bind("<Return>", _on_enter_pressed)

# # send button
# send_button2 = Button(bottom_label2, text="send", font="FONT_BOLD", width=20, bg=BG_GRAY,
#                       command=lambda: _on_enter_pressed(frame2))
# send_button2.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

# frame2.place(relx=0, rely=0.13, relheight=0.87, relwidth=1)

