import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
# client = OpenAI()
# OpenAI API 키 설정 (실제 사용 시 본인의 API 키로 대체 필요)
client = OpenAI(api_key="")

def collect_training_data():
    # human_texts 2개 이상
    human_texts = ["""
위 코드 예시는 OpenAI API를 활용해 “killer prompt”를 포함한 간단한 분류 함수를 구현한 것입니다.

    detect_ai_generated(text) 함수는 입력된 텍스트를 모델에 전달하여 AI-generated 또는 Human-written 레이블과 함께 간략한 근거를 반환합니다.

    --file 또는 --text 인자를 통해 분석할 텍스트를 지정할 수 있으며, 결과를 콘솔에 출력하도록 되어 있습니다.
""",
"""
CA 번들 갱신: certifi.where()를 사용하거나 시스템 CA를 업데이트하여 인증서 문제를 줄일 수 있습니다.

Selenium의 네트워크 인터셉트 활용: 크롬 DevTools Protocol(CDP)을 통해 이미지 바이너리를 직접 가져오는 방법도 있습니다.

보안: 프로덕션에서는 verify=True와 올바른 CA 번들을 유지하도록 권장합니다.
""",
"""
process_page 함수 내부에 iframe 전환 로직을 추가하고, 전환 성공 시 해당 iframe의 내용을 크롤링하도록 전체 코드를 수정했습니다. 주요 변경 사항은 다음과 같습니다:

selenium.webdriver.common.by, WebDriverWait, expected_conditions 모듈 임포트

process_page 시작 부분에서 By.ID("mainFrame")으로 iframe 전환 시도

iframe 전환 성공 시 driver.page_source를 가져온 후 driver.switch_to.default_content() 호출로 기본 컨텍스트로 복귀
"""

] # 인간이 작성한 텍스트 샘플
    
    ai_texts = []
    for prompt in ["자기소개서 작성해줘", "에세이 작성해줘"]:
    # for prompt in ["자기소개서 작성해줘", "블로그 게시물 작성해줘", "에세이 작성해줘"]:
        response = client.chat.completions.create(
            model="gpt-4",                         
            messages=[                            
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user",   "content": prompt}
            ],
            max_tokens=500
        )
        ai_texts.append(response.choices[0].message.content)
    
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    all_texts = human_texts + ai_texts
    
    return all_texts, labels

def train_detector():
    texts, labels = collect_training_data()
    
    # 텍스트를 특성 벡터로 변환
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # 모델 훈련
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, vectorizer

def detect_ai_text(text, model, vectorizer):
    # 텍스트를 특성 벡터로 변환
    X = vectorizer.transform([text])
    
    # 예측
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction, probability

# 모델 훈련
model, vectorizer = train_detector()

# 예시 텍스트 감지
example_text = "여기에 분석하고 싶은 텍스트를 입력하세요."
is_ai, probability = detect_ai_text(example_text, model, vectorizer)

print(f"AI 텍스트 여부: {'예' if is_ai else '아니오'}")
print(f"AI 작성 확률: {probability:.2f}")
#  답변 예시 
# AI 텍스트 여부: 예
# AI 작성 확률: 0.66