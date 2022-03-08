import pandas as pd
from eunjeon import Mecab
import datetime
from datetime import date
from bs4 import BeautifulSoup
from urllib import parse
import Model
import numpy as np
import requests
import random
from GPT3tutorial import gpt3

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# model = Model.KoNET(firstTraining=True, testCase=False)
# model.model_setting()

text_casual = pd.read_csv('DBD/conversation_QA.csv')
text_knowledge = pd.read_csv('DBD/conversation_QA_park.csv')
text_restaurant = pd.read_csv('DBD/restaurant_list.csv')
text_spot = pd.read_csv('DBD/spot_list.csv')

mecab = Mecab()

def softmax(probs):
    exp_probs = np.exp(probs)
    sum_exp_probs = np.sum(exp_probs)
    return exp_probs / sum_exp_probs

def similarity(bow, bow1, bow2):
    cnt = 0
    for token in bow1:
        if token in bow2:
            cnt = cnt + 1
    for token2 in bow:
        if token2 in bow2:
            cnt = cnt + 3
    return cnt / (len(bow1) + len(bow2))

'''
def similarity(sen1, sen2):
    sentence = (sen1, sen2)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity
'''

def find_index(num, data):
    index = []
    for i in range(len(data)):
        if data[i] == num:
            index.append(i)
    return index

def makeanswer(query):
    query = query.strip('!')
    query = query.strip('.')
    query = query.strip(',')
    query = query.strip('?')

    bow1 = mecab.morphs(query)
    bow = mecab.nouns(query)

    strtime = datetime.datetime.now().strftime("%H:%M:%S")
    Time = strtime.split(':')
    today = date.today()
    whatday = today.weekday()

    if (query.__contains__('몇시') and query.__contains__('지금')) or (
            query.__contains__('시간') and query.__contains__('지금')) or (
            query.__contains__('시간') and query.__contains__('현재')
            or (query.__contains__('시간') and query.__contains__('알려줘'))):
        if (query.__contains__('휠체어') or query.__contains__('유모차')):
            answer = '유모차와 휠체어는 오전 9시부터 오후 6시까지 대여할 수 있습니다.'
        elif (query.__contains__('에스컬레이터')):
            answer = '에스컬레이터는 매일 24시간 운영됩니다.'
        else:
            if int(Time[0]) > 12:
                Time[0] = int(Time[0]) - 12
                day = "오후"
            else:
                day = "오전"
            answer = f'지금 시간은 {day} {Time[0]}시 {Time[1]}분입니다.'

    # 인사
    elif (query.__contains__('안녕')):
        answer = "안녕하세요! 만나서 반가워요."

    elif (query.__contains__('고마워') or query.__contains__('고맙') or query.__contains__('감사')):
        answer = "별말씀을요!"

    elif (query.__contains__('반가워') or query.__contains__('반갑습니다')):
        answer = "저도 반가워요!"

    elif (query.__contains__('상담원') or query.__contains__('오퍼레이터')):
        answer = "상담원을 연결합니다."

    # 날씨
    elif ((query.__contains__('날씨') and query.__contains__('알려줘')) or (
            query.__contains__('날씨') and query.__contains__('어때')) or (
                  query.__contains__('날씨') and query.__contains__('뭐야'))):

        url = 'https://search.naver.com/search.naver?ie=UTF-8&sm=whl_hty&query=%EC%98%A4%EB%8A%98+%EC%A4%91%EA%B5%AC+%EB%82%A8%ED%8F%AC%EB%8F%99+%EB%82%A0%EC%94%A8'
        headers = {"User-Agent": "mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/90.0.4430.232 whale/2.10.124.26 safari/537.36"}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'lxml')

        weather = soup.find("span", attrs={"class": "weather before_slash"}).get_text()
        temp_now = soup.find("div", attrs={"class": "temperature_text"}).get_text().replace("현재 온도", "")
        temp_low = soup.find("span", attrs={"lowest"}).get_text().replace("최저기온", "")
        temp_high = soup.find("span", attrs={"highest"}).get_text().replace("최고기온", "")

        # dust = soup.find("ul", attrs={"today_chart_list"})
        # dust1 = dust.find_all("li")[0].get_text().strip().replace("미세먼지", "")
        # sun = dust.find_all("li")[2].get_text().strip().replace("자외선", "")

        # sunset = dust.find_all("li")[3].get_text().strip().replace("일몰", "")
        # time = sunset.split(":")

        # if int(time[0]) > 12:
        #    time[0] = int(time[0]) - 12

        answer = f"현재 날씨는 {weather}이며 기온은 {temp_now}입니다. 오늘 최저 기온은 {temp_low}, 최고 기온은 {temp_high}입니다. "

    # 길찾기
    elif (query.__contains__('길') and query.__contains__('알려줘')) or (query.__contains__('어떻게가')) or (
            query.__contains__('어디에있어')) or (query.__contains__('어디야') or (query.__contains__('어디있어'))):
        # start는 virtual agent놓이는 위치따라 바뀌어야 함\
        # 지금은 용두산공원 이순신장군 동상 입구
        start = '14363937.949299295,4177284.0455231294'
        start = parse.quote(start)

        if (query.__contains__('종각') or query.__contains__('꽃시계')):
            destination = '종각'

        elif (query.__contains__('면세점')):
            destination = '면세점'

        elif (query.__contains__('정수사') or query.__contains__('사찰')):
            destination = '사찰'

        elif (query.__contains__('광장') or query.__contains__('중앙광장')):
            destination = "광장"

        elif (query.__contains__('화장실')):
            destination = "화장실"

        elif (query.__contains__('충무공동상') or query.__contains__('이순신장군동상')):
            destination = '충무공동상'

        elif (query.__contains__('팔각정')):
            destination = '팔각정'

        elif (query.__contains__('부산탑') or query.__contains__('전망대') or query.__contains__('부산타워')):
            destination = '타워'

        elif (query.__contains__('공영주차장') or query.__contains__('주차장')):
            destination = '주차장'

        elif (query.__contains__('쉼터') or query.__contains__('벤치')):
            destination = '쉼터'

        elif (query.__contains__('관리사무소') or query.__contains__('관리실')):
            destination = '관리실'

        elif (query.__contains__('한복대여샵') or query.__contains__('아담') or query.__contains__('한복남')):
            destination = '아담'

        elif (query.__contains__('영화체험관') or query.__contains__('영화체험박물관')):
            destination = '영화'

        elif (query.__contains__('트릭아이') or query.__contains__('트릭아이뮤지엄')):
            destination = '트릭아이'

        elif (query.__contains__('편의점') or query.__contains__('마트')):
            destination = '편의점'

        elif (query.__contains__('선박전시') or query.__contains__('모형선박')):
            destination = '선박'

        elif (query.__contains__('40계단')):
            destination = '계단'

        elif (query.__contains__('비프광장') or query.__contains__('비프거리')):
            destination = '비프'

        elif (query.__contains__('국제시장')):
            destination = '국제'

        elif (query.__contains__('자갈치')):
            destination = '자갈치'

        elif (query.__contains__('보건소') or query.__contains__('병원') or query.__contains__('응급실')):
            destination = "보건소"

        else:
            answer = 'tssss'
            # webbrowser.open_new_tab(URL)
            return answer

        answer = 'h' + destination

    # 편의시설(유모차, 휠체어,에스컬레이터)
    elif (query.__contains__('유모차') or query.__contains__('휠체어')):
        if (query.__contains__('어디')):
            destination = '유모차'
            answer = destination

        elif (query.__contains__('전화번호')):
            answer = '전화번호는 051-860-7820 입니다.'

    elif (query.__contains__('에스컬레이터')):
        if query.__contains__('어디'):
            answer = '에스컬레이터는 바로 앞에 위치해있습니다.'

    # 사진 스팟
    elif (query.__contains__('사진') or query.__contains__('포토')):
        answer = ['팔각정에서 부산항 대교를 배경으로 사진 찍어보는 거 어때요?',
                  '사랑의 자물쇠 앞에서 사진이 잘 나와요!']
        answer = random.choice(answer)
    
    # 맛집 추천
    elif (query.__contains__('맛집') or query.__contains__('먹거리') or query.__contains__('식당') or query.__contains__('밥집') or query.__contains__('먹')):
        remove_list = ['맛집', '먹거리', '식당', '밥집']
        for i in range(bow.__len__()):
            for j in range(remove_list.__len__()):
                if bow[i-1] == remove_list[j-1]:
                    bow.remove(remove_list[j-1])

        rest_list = []
        for i in range(text_restaurant.__len__()):
            for j in range(bow.__len__()):
                if text_restaurant['location'][i].__contains__(bow[j]):
                    if ((text_restaurant['closedate1'][i] != whatday) and (
                            text_restaurant['closedate2'][i] != whatday) and (
                            text_restaurant['closedate3'][i] != whatday)):
                        if ((int(Time[0]) > int(text_restaurant['opentime1'][i])) and (int(Time[0]) < int(text_restaurant['closetime1'][i]))):
                            rest_list.append(text_restaurant['name'][i])
                        elif (int(Time[0]) == int(text_restaurant['opentime1'][i])):
                            if (int(Time[1]) >= int(text_restaurant['opentime2'][i])):
                                rest_list.append(text_restaurant['name'][i])
                        elif (int(Time[0]) == int(text_restaurant['closetime1'][i])):
                            if (int(Time[1]) <= int(text_restaurant['closetime2'][i])):
                                rest_list.append(text_restaurant['name'][i])
        if len(rest_list) >= 2:
            recom = random.sample(rest_list, 2)
            answer = f'부산광역시 보건위생과에서 선정한 식당인 {recom[0]} 혹은 {recom[1]} 어때요?'
        elif len(rest_list) == 1:
            recom = rest_list
            answer = f'부산광역시 보건위생과에서 선정한 식당인 {recom[0]} 어때요?'
        elif len(rest_list) == 0:
            for i in range(len(text_restaurant)):
                if text_restaurant['location'][i].__contains__('중구') or text_restaurant['location'][i].__contains__('영도구'):
                    rest_list.append(text_restaurant['name'][i])
            recom = random.sample(rest_list, 2)
            answer = f'부산광역시 보건위생과에서 선정한 식당인 {recom[0]} 혹은 {recom[1]} 어때요?'

    # 근처 명소 추천
    elif (query.__contains__('명소') or query.__contains__(('놀거리')) or query.__contains__('놀만한') or query.__contains__('가볼만한') or query.__contains__('갈만한') or query.__contains__('놀데')):
        spot_list = []
        for i in range(text_spot.__len__()):
            if query.__contains__(text_spot['category'][i]):
                spot_list.append(text_spot['name'][i])
            else:
                for j in range(bow.__len__()):
                    if text_spot['location'][i].__contains__(bow[j]):
                        if ((text_spot['closedate1'][i] != whatday) and (text_spot['closedate2'][i] != whatday)):
                            if ((int(Time[0]) > int(text_spot['opentime1'][i])) and (
                                    int(Time[0]) < int(text_spot['closetime1'][i]))):
                                spot_list.append(text_spot['name'][i])
                            elif (int(Time[0]) == int(text_spot['opentime1'][i])):
                                if (int(Time[1]) >= int(text_spot['opentime2'][i])):
                                    spot_list.append(text_spot['name'][i])
                            elif (int(Time[0]) == int(text_spot['closetime1'][i])):
                                if (int(Time[1]) <= int(text_spot['closetime2'][i])):
                                    spot_list.append(text_spot['name'][i])

        if len(spot_list) >= 2:
            recom = random.sample(spot_list, 2)
            answer = f'부산 관광 문화청에서 선정한 부산 명소인 {recom[0]} 혹은 {recom[1]} 어때요?'
        elif len(spot_list) == 1:
            recom = spot_list
            answer = f'부산 관광 문화청에서 선정한 부산 명소인 {recom[0]} 어때요?'
        elif len(spot_list) == 0:
            for i in range(text_spot.__len__()):
                if text_spot['location'][i].__contains__('중구') or text_spot['location'][i].__contains__('영도구'):
                    spot_list.append(text_spot['name'][i])
            recom = random.sample(spot_list, 2)
            answer = f'부산 관광 문화청에서 선정한 부산 명소인 {recom[0]} 혹은 {recom[1]} 어때요?'

    else:
        answer = gpt3(query)

        # [0,0]이 끊기지 않을 확률, [0,1]이 끊길 확률이니까
        # 근데 지금 내가 .. 데이터셋을 반대로 학습시켜버렸으니까..
        # if softmaxed_probs[0, 0] <= softmaxed_probs[0, 1]: 이건 원래 데이터셋

        '''    
        if softmaxed_probs[0, 0] >= softmaxed_probs[0, 1]:
            answer = "-1" + answer

        else:
            answer = answer
        '''
    return answer