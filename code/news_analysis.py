import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# ==========================================
# 1. 설정 (User Configuration)
# ==========================================
INPUT_FILE = "./raw_data/NewsResult_20150101-20250331_KTX코레일_KK.xlsx"
OUTPUT_FILE = "./processed_data/KTX_Monthly_Demand_Forecast_Data.csv"
# 허깅페이스 토큰 (본인의 토큰을 입력하세요)
HF_TOKEN = "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" 
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 주요 분석 카테고리 (감성 분석 결과를 남길 분야)
TARGET_CATEGORIES = ['경제', '사회', '지역', '정치'] 

# ==========================================
# 2. 데이터 로드 및 전처리 함수
# ==========================================
def clean_text(text):
    """특수문자 제거 및 텍스트 정제"""
    if not isinstance(text, str): return ""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text) 
    return re.sub(r'\s+', ' ', text).strip()

print("1. 데이터 로드 중...")
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {INPUT_FILE}")
    exit()

# 날짜 변환
df['일자'] = pd.to_datetime(df['일자'], format='%Y%m%d')

# 텍스트 합치기 및 정제
df['Processed_Text'] = (df['제목'] + " " + df['본문']).apply(clean_text)

# 카테고리 대분류 추출
df['Main_Category'] = df['통합 분류1'].astype(str).apply(lambda x: x.split('>')[0])

print(f"   - 총 데이터 건수: {len(df)}건")
print("   - 전처리 완료.")

# ==========================================
# 3. Llama 모델 로드 및 감성 분석
# ==========================================
print("\n2. Llama 모델 로드 중 (GPU 필요)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )
    print("   - 모델 로드 성공!")

except Exception as e:
    print(f"   - 모델 로드 실패: {e}")
    text_generator = None

def get_llama_sentiment(text):
    """Llama에게 긍정(1)/부정(-1)/중립(0) 판단 요청"""
    if text_generator is None:
        return np.random.choice([-1, 0, 1]) 

    short_text = text[:400] 
    
    messages = [
        {"role": "system", "content": "Analyze the sentiment of the news regarding KTX demand. Reply with 'Positive' if it increases demand, 'Negative' if it decreases demand (like accidents, price hikes), or 'Neutral'."},
        {"role": "user", "content": f"News: {short_text}\n\nSentiment (Positive/Negative/Neutral):"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        outputs = text_generator(prompt)
        result = outputs[0]['generated_text'].lower()
        
        if 'negative' in result: return -1.0
        elif 'positive' in result: return 1.0
        else: return 0.0
    except:
        return 0.0

print("\n3. 뉴스 감성 분석 시작 (시간이 소요됩니다)...")
tqdm.pandas()
df['Sentiment_Score'] = df['Processed_Text'].progress_apply(get_llama_sentiment)

# ==========================================
# 4. 월별 데이터 집계 (수정됨: Count 컬럼 삭제)
# ==========================================
print("\n4. 월별 데이터 집계 중...")

# 4-1. 기본 월별 집계 (전체 뉴스 수, 전체 평균 감성)
# 전체 뉴스량(Total_News_Count)은 수요 예측에 중요할 수 있어 남겨두었습니다.
df_monthly_base = df.set_index('일자').resample('M').agg({
    'Processed_Text': 'count',       # 전체 뉴스 보도량
    'Sentiment_Score': 'mean'        # 전체 뉴스 감성 평균
}).rename(columns={'Processed_Text': 'Total_News_Count', 'Sentiment_Score': 'Total_Sentiment_Mean'})

# 4-2. 카테고리별 감성 점수 (Sentiment by Category)
# (이전 코드의 'Count_' 생성 부분은 삭제했습니다)
df['Month'] = df['일자'].dt.to_period('M')

df_monthly_sentiments = df.pivot_table(
    index='Month',
    columns='Main_Category',
    values='Sentiment_Score',
    aggfunc='mean'
).fillna(0) # 뉴스가 없으면 0(중립)

# 컬럼명 변경 (Sent_경제, Sent_사회...) 및 타겟 카테고리만 선택
df_monthly_sentiments.columns = [f'Sent_{col}' for col in df_monthly_sentiments.columns]
selected_sent_cols = [f'Sent_{c}' for c in TARGET_CATEGORIES if f'Sent_{c}' in df_monthly_sentiments.columns]
df_monthly_sentiments = df_monthly_sentiments[selected_sent_cols]

# 인덱스를 타임스탬프로 변환
df_monthly_sentiments.index = df_monthly_sentiments.index.to_timestamp(freq='M')

# ==========================================
# 5. 최종 병합 및 저장
# ==========================================
final_df = pd.concat([
    df_monthly_base,
    df_monthly_sentiments
], axis=1).fillna(0)

# 인덱스 날짜 포맷 정리 (2025-01)
final_df.index = final_df.index.strftime('%Y-%m')

# [중요] 인덱스 이름을 'Date'로 지정
final_df.index.name = 'Date'

print("\n[미리보기] 최종 데이터 셋:")
print(final_df.head())

final_df.to_csv(OUTPUT_FILE)
print(f"\n완료! '{OUTPUT_FILE}' 파일에 저장되었습니다.")