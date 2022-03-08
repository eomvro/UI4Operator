import requests

def get_translate_ko_en(text):
    data = {'text' : text,'source' : 'ko','target' : 'en'}

    client_id = "gxbfXWoQIE0OfT1_V1kR"
    client_secret = "9EMiLxKgZS"
    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data= data)
    rescode = response.status_code

    if(rescode==200):
        t_data = response.json()
        return t_data['message']['result']['translatedText']
    else:
        print("Error Code:" , rescode)
