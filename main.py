from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Inisialisasi FastAPI
app = FastAPI()

# Load model dan tokenizer
model = load_model('gambling_detection_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Maksimal panjang sequence (harus sesuai dengan model training)
maxlen = 100

# Fungsi pembersihan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    text = text.lower()  
    return text

# Fungsi prediksi teks
def predict_text(text):
    # Preprocessing teks (bersihkan teks)
    text_cleaned = clean_text(text)
    
    # Tokenisasi dan padding
    sequence = tokenizer.texts_to_sequences([text_cleaned])  # Tokenisasi teks
    padded = pad_sequences(sequence, maxlen=maxlen)  # Padding sequence
    
    # Prediksi menggunakan model
    prediction = model.predict(padded)[0][0]
    
    # Kembalikan hasil
    return {
        'text': text,
        'probability': float(prediction),
        'prediction': 'Judi Online' if prediction > 0.5 else 'Bukan Judi Online'
    }

# Endpoint untuk prediksi
@app.post("/predict")
def predict_endpoint(text: str):
    try:
        # Gunakan fungsi predict_text untuk memprediksi input
        result = predict_text(text)
        return result
    except Exception as e:
        # Tangani error jika terjadi masalah
        raise HTTPException(status_code=500, detail=str(e))
