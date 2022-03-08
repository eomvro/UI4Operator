query = '요즘영화추천해줘'
query = query.split('추천')
print(query[0])

# 맛집
elif (query.__contains__('맛집') or query.__contains__('밥집')):
webbrowser.open_new(
    'https://map.naver.com/v5/search/%EB%B6%80%EC%82%B0%EB%8C%80%EB%A7%9B%EC%A7%91?c=14369210.1405708,4194961.1475482,15,0,0,0,dh')
answer = "한 번 골라보세요."

# 영화
elif (query.__contains__('영화')):
webbrowser.open_new_tab(
    'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%83%81%EC%98%81+%EC%98%81%ED%99%94')
answer = "한 번 골라보세요."

# 드라마
elif (query.__contains__('드라마')):
webbrowser.open_new_tab(
    'https://search.naver.com/search.naver?where=nexearch&sm=top_sly.hst&fbm=1&acr=1&ie=utf8&query=%EB%93%9C%EB%9D%BC%EB%A7%88')
answer = "한 번 골라보세요."

# 예능
elif (query.__contains__('예능')):
webbrowser.open_new_tab(
    'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%98%88%EB%8A%A5')
answer = "한 번 골라보세요."