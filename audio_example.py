import speech_recognition as sr
from makeanswer2 import makeanswer
r = sr.Recognizer()

print('대기중')
while True:
    try:
        with sr.Microphone() as source:
            audio = r.listen(source, None, 5)
            question = r.recognize_google(audio, language='ko-KR')
            print('질문 : ', question)
            answer = makeanswer(question)
            print('대답 : ', answer)
    except sr.UnknownValueError as e:
        pass