import requests
client_id = "gVHHjDqJ4ozBqkkKdZD5"
client_secret = "DzLw8onjl1"

# 유명인 얼굴인식
url = "https://openapi.naver.com/v1/vision/celebrity"
files = {'image': open('face.jpg', 'rb')}
headers = {'X-Naver-Client-Id': client_id, 'X-Naver-Client-Secret': client_secret}
response = requests.post(url,  files=files, headers=headers)
rescode = response.status_code

text = response.text
text = text.split('"')
text = text[17]
answer = text + ' 닮았어요.'

if(rescode==200):
    print(answer)
else:
    print('다른 사진 보여줭')