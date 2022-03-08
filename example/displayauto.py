from bs4 import BeautifulSoup
from urllib import parse, request
import datetime
import pyautogui
import webbrowser

#검색
d="고양이"
d=parse.quote(d)

print(d)
URL='https://www.youtube.com/results?search_query='+d
webbrowser.open_new_tab(URL)

# 좌표 객체 얻기
position = pyautogui.position()

# 화면 전체 크기 확인하기
print(pyautogui.size())

# x, y 좌표
print(position.x)
print(position.y)

# 마우스 이동 (x 좌표, y 좌표)
pyautogui.moveTo(900, 380)

# 마우스 이동 (x 좌표, y 좌표 2초간)
#pyautogui.moveTo(100, 100, 2)

# 마우스 이동 ( 현재위치에서 )
pyautogui.moveRel(0, 0, 2)

# 마우스 클릭
pyautogui.click()

# 2초 간격으로 2번 클릭
#pyautogui.click(clicks= 2, interval=2)

# 더블 클릭
#pyautogui.doubleClick()

# 오른쪽 클릭
#pyautogui.click(button='right')

# 스크롤하기
#pyautogui.scroll(10)

# 드래그하기
#pyautogui.drag(0, 300, 1, button='left')


