#GPT 3 tutorial

import openai
import googletrans
import requests

def gpt3(query):
    openai.api_key = "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"
    #translated_query = get_translate_ko_en(query)
    response = openai.Completion.create(
        engine = "davinci-instruct-beta",
        prompt = query,
        temperature = 0.1,
        max_tokens = 1000,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    answer = response.choices[0].text.split('.')[0]
    #translated_answer = get_translate_en_ko(answer)
    return answer


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
        #return t_data
        return t_data['message']['result']['translatedText']
    else:
        print("Error Code:" , rescode)

import requests

def get_translate_en_ko(text):
    data = {'text' : text,'source' : 'en','target' : 'ko'}

    client_id = "gxbfXWoQIE0OfT1_V1kR"
    client_secret = "9EMiLxKgZS"
    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data= data)
    rescode = response.status_code

    if(rescode==200):
        t_data = response.json()
        #return t_data
        return t_data['message']['result']['translatedText']
    else:
        print("Error Code:" , rescode)