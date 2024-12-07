from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

app = FastAPI()

MAX_WORDS = 10000
MAX_LEN = 250

class URLRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def scrape_web_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')  
        text = soup.get_text(separator=" ", strip=True)
        processed_text = preprocess_text(text)

        
        return processed_text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scraping URL: {str(e)}")

def load_model_and_tokenizer():
    try:
        model_web = load_model('./models/modelLajuA.keras')
        model_text = load_model('./models/modelLajuB.keras')
        with open('./models/tokenizerA.json', 'r', encoding='utf-8') as f:
            tokenizer_web_config = f.read()
        with open('./models/tokenizerB.json', 'r', encoding='utf-8') as f:
            tokenizer_text_config = f.read()
        
        tokenizer_web = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_web_config)
        tokenizer_text = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_text_config)
        
        return model_web, model_text, tokenizer_web, tokenizer_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

MODEL_WEB, MODEL_TEXT, TOKENIZER_WEB, TOKENIZER_TEXT = load_model_and_tokenizer()

@app.post("/web")
async def predict_web_content(request: URLRequest):
    try:
        web_text = scrape_web_content(request.url)
        web_sequence = TOKENIZER_WEB.texts_to_sequences([web_text])
        web_padded = pad_sequences(web_sequence, maxlen=MAX_LEN)
        prediction = MODEL_WEB.predict(web_padded)
        probability = prediction[0][0]
        
        return {
            "url": request.url,
            "is_judi_online": bool(probability > 0.5),
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/iklan")
async def predict_text(request: TextRequest):
    try:
        processed_text = preprocess_text(request.text)
        text_sequence = TOKENIZER_TEXT.texts_to_sequences([processed_text])
        text_padded = pad_sequences(text_sequence, maxlen=MAX_LEN)
        prediction = MODEL_TEXT.predict(text_padded)
        probability = prediction[0][0]
        
        return {
            "text": request.text,
            "is_judi_online": bool(probability > 0.5),
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))