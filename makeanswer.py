import pandas as pd
from eunjeon import Mecab
import datetime
from bs4 import BeautifulSoup
from urllib import parse
import Model
import numpy as np
import requests
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

model = Model.KoNET(firstTraining=True, testCase=False)
model.model_setting()

text_casual = pd.read_csv('DBD/conversation_QA.csv')
text_knowledge = pd.read_csv('DBD/conversation_QA_park.csv')

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

    # 현재 시간
    if (query.__contains__('몇시') and query.__contains__('지금')) or (
            query.__contains__('시간') and query.__contains__('지금')) or (
            query.__contains__('시간') and query.__contains__('현재')
            or (query.__contains__('시간') and query.__contains__('알려줘'))):
        if (query.__contains__('휠체어') or query.__contains__('유모차')):
            answer = '유모차와 휠체어는 오전 9시부터 오후 6시까지 대여할 수 있습니다.'
        elif (query.__contains__('에스컬레이터')):
            answer = '에스컬레이터는 매일 24시간 운영됩니다.'
        else:
            strtime = datetime.datetime.now().strftime("%H:%M:%S")
            Time = strtime.split(':')
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


    # 홈페이지 틀어주기
    elif (query.__contains__('지도')):
        URL = 'm' + 'https://map.naver.com/v5/entry/place/11622430?c=14363777.0592392,4177580.6877232,17,0,0,0,dh'
        answer = URL

    # 날씨
    elif ((query.__contains__('날씨') and query.__contains__('알려줘')) or (
            query.__contains__('날씨') and query.__contains__('어때')) or (
                  query.__contains__('날씨') and query.__contains__('뭐야'))):

        url = 'https://search.naver.com/search.naver?ie=UTF-8&sm=whl_hty&query=%EC%98%A4%EB%8A%98+%EC%A4%91%EA%B5%AC+%EB%82%A8%ED%8F%AC%EB%8F%99+%EB%82%A0%EC%94%A8'
        headers = {
            "User-Agent": "mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/90.0.4430.232 whale/2.10.124.26 safari/537.36"}
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

    # 지도
    elif ((query.__contains__('지도') and query.__contains__('틀어')) or (
            query.__contains__('지도') and query.__contains__('보여'))):
        URL = "https://map.naver.com/v5/?c=14363849.0384220,4177568.6052696,18,0,0,0,dh"
        # webbrowser.open_new_tab(URL)
        answer = URL

    # 영상(유튜브)틀기
    elif ((query.__contains__('영상') and query.__contains__('틀어줘')) or (
            query.__contains__('영상') and query.__contains__('보여줘'))):
        query = query.strip("영상틀어줘")
        query = query.strip("영상보여줘")
        query = query.strip('유튜브')
        query = query.strip('유튜브에')
        query = query.strip('유튜브에서')
        query = parse.quote(query)
        URL = "v" + "https://www.youtube.com/results?search_query=" + query
        answer = URL

    # 길찾기
    elif (query.__contains__('길') and query.__contains__('알려줘')) or (query.__contains__('어떻게가')) or (
            query.__contains__('어디에있어')) or (query.__contains__('어디야') or (query.__contains__('어디있어'))):
        # start는 virtual agent놓이는 위치따라 바뀌어야 함\
        # 지금은 용두산공원 이순신장군 동상 입구
        start = '14363937.949299295,4177284.0455231294, PLACE_POI'
        start = parse.quote(start)

        if (query.__contains__('종각') or query.__contains__('꽃시계')):
            destination = '14363910.119426597,4177501.566940369, PLACE_POI'

        elif (query.__contains__('면세점')):
            destination = '14363763.2667543,4177443.7676264,PLACE_POI'

        elif (query.__contains__('정수사') or query.__contains__('사찰')):
            destination = '/14363830.269955847,4177507.145515334,PLACE_POI'

        elif (query.__contains__('광장') or query.__contains__('중앙광장')):
            destination = '14363825.50548165,4177566.904474849,PLACE_POI'

        elif (query.__contains__('화장실')):
            destination1 = '14363804.866848055,4177556.5500419065,PLACE_POI'
            destination2 = '14363922.809848543,4177544.9710488506,PLACE_POI'
            destination3 = '14363990.113612678,4177472.0549648646,PLACE_POI'
            des = [destination1, destination2, destination3]
            destination = random.choice(des)

        elif (query.__contains__('충무공동상') or query.__contains__('이순신장군동상')):
            destination = '14363863.7548587,4177540.4673545,PLACE_POI'

        elif (query.__contains__('팔각정')):
            destination = '14363840.121730786,4177644.896388078,PLACE_POI'

        elif (query.__contains__('부산탑') or query.__contains__('전망대') or query.__contains__('부산타워')):
            destination = '14363815.586915012,4177625.316719787,PLACE_POI'

        elif (query.__contains__('공영주차장') or query.__contains__('주차장')):
            destination = '14363908.739064913,4177809.6581486836,PLACE_POI'

        elif (query.__contains__('쉼터') or query.__contains__('벤치')):
            destination = '14363908.906044144,4177635.6848233603,PLACE_POI'

        elif (query.__contains__('관리사무소')):
            destination = '14363990.113612678,4177472.0549648646,PLACE_POI'

        elif (query.__contains__('한복대여샵') or query.__contains__('아담') or query.__contains__('한복남')):
            destination = '14363825.928495709,4177606.594289588,PLACE_POI'

        elif (query.__contains__('영화체험관') or query.__contains__('영화체험박물관')):
            destination = '14363969.753277812,4177701.227245692,PLACE_POI'

        elif (query.__contains__('트릭아이') or query.__contains__('트릭아이뮤지엄')):
            destination = '14363937.648736667,4177727.0252412073,PLACE_POI'

        elif (query.__contains__('편의점') or query.__contains__('마트')):
            destination1 = '14364075.306418981,4177275.256044045,PLACE_POI'
            destination2 = '14364031.446539614,4177341.204519095,PLACE_POI'
            des = [destination1, destination2]
            destination = random.choice(des)

        elif (query.__contains__('선박전시') or query.__contains__('모형선박')):
            destination = '14363824.210236836,4177594.1212122794,PLACE_POI'

        elif (query.__contains__('40계단')):
            destination = '14364112.665240098,4178020.100669941,PLACE_POI'

        elif (query.__contains__('비프광장') or query.__contains__('비프거리')):
            destination = '14363303.439333718,4177312.1419697013,PLACE_POI'

        elif (query.__contains__('국제시장')):
            destination = '14363339.312838655,4177632.339726424,PLACE_POI'

        elif (query.__contains__('자갈치')):
            destination = '14363677.483954739,4177033.425647039,PLACE_POI'

        elif (query.__contains__('보건소') or query.__contains__('병원') or query.__contains__('응급실')):
            destination1 = '14363729.470156942,4177776.0225487407'
            destination2 = '14363941.255488168,4177837.7695333567'
            place = [destination1, destination2]
            destination = random.choice(place)

        else:
            answer = 't' + 'https://map.naver.com/v5/?c=14363803.9985560,4177553.3389451,18,0,0,0,dh'
            # webbrowser.open_new_tab(URL)
            return answer

        destination = parse.quote(destination)
        URL = 'https://map.naver.com/v5/directions/' + start + '/' + destination + '/-/walk?c=14368852.8495332,4195776.3347758,16,0,0,0,dh'
        answer = URL

    # 편의시설(유모차, 휠체어,에스컬레이터)
    elif (query.__contains__('유모차') or query.__contains__('휠체어')):
        if (query.__contains__('어디')):
            destination = '14363990.113612678,4177472.0549648646,PLACE_POI'
            start = '14363937.949299295,4177284.0455231294, PLACE_POI'
            start = parse.quote(start)
            destination = parse.quote(destination)
            URL = 'https://map.naver.com/v5/directions/' + start + '/' + destination + '/-/walk?c=14368852.8495332,4195776.3347758,16,0,0,0,dh'
            answer = URL

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

    # 근처 맛집, 명소 추천
    elif (query.__contains__('추천')):
        if (query.__contains__('명소') or query.__contains__('관광지') or query.__contains__("놀거리")):
            placelist = ['부산타워 전망대', '부산 영화 체험 박물관', '트릭아이 뮤지엄', '세계 모형 선박 전시관', '비프 광장', '국제 시장', '자갈치 시장',
                         '세계 민속 악기 박물관']
            place = random.sample(placelist, 2)
            answer = place[0] + ' 혹은 ' + place[1] + ' 추천할게요!'

        else:
            query = query.strip('남포동')
            query = query.strip('여기')
            query = query.strip("에")
            query = query.strip("근처")
            query = query.strip('이')
            query = query.strip('요')
            query = query.strip('용두산공원')
            query = query.split('추천')

            query = query[0]
            query = '남포동' + query
            query = parse.quote(query)
            url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=' + query
            headers = {
                "User-Agent": "mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/90.0.4430.232 whale/2.10.124.26 safari/537.36"}
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'lxml')
            meallist = soup.find("ul", attrs={"class": "_3bohv _1V_Nc"})
            mlist = []
            meal = []

            for i in range(6):
                mlist.append(meallist.find_all("li")[i])
                if mlist[i].find_all("span")[0].get_text() == '광고':
                    meal.append(mlist[i].find_all("span")[1].get_text())
                else:
                    meal.append(mlist[i].find_all("span")[0].get_text())

            recom = random.sample(meal, 2)
            answer = recom[0] + ' 혹은 ' + recom[1] + ' 어때요?'
    else:
        similar1 = []
        for i in range(len(text_casual)):
            bow2 = mecab.morphs(text_casual['Q'][i])
            similar1.append(similarity(bow, bow1, bow2))
        num1 = find_index(max(similar1), similar1)
        number1 = random.choice(num1)
        answer = text_casual['A'][number1]

        probs = model.make_propagate(query, answer)
        # probs[0, 0] = abs(probs[0, 0])
        # probs[0, 1] = abs(probs[0, 1])
        softmaxed_probs = softmax(probs)

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
