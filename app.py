import joblib
import os
import re
from flask import Flask, request, jsonify, render_template

# ==============================================================================
# Inisialisasi Aplikasi Flask
# ==============================================================================
app = Flask(__name__)


# ==============================================================================
# 1. MEMUAT MODEL DAN KAMUS
# ==============================================================================
try:
    model = joblib.load('best_model_svc.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("="*50)
    print("Model ML dan Vectorizer berhasil dimuat.")
except FileNotFoundError:
    print("Peringatan: File model untuk endpoint /predict tidak ditemukan.")
    model = None
    vectorizer = None

aspect_keywords = {
    'Fasilitas': ['fasilitas', 'lokasi', 'kamar', 'kamar mandi', 'ac', 'tv', 'wifi', 'kolam renang', 'parkir', 'transportasi', 'restoran', 'makanan', 'menu', 'sarapan', 'lengkap'],
    'Staf': ['staff', 'staf', 'pelayanan', 'karyawan', 'resepsionis', 'petugas', 'ramah', 'sopan', 'membantu', 'gercep', 'sigap', 'respon', 'lambat', 'jutek','layanan'],
    'Kebersihan': ['bersih', 'kebersihan', 'rapi', 'wangi', 'nyaman', 'kotor', 'bau', 'debu', 'sprei', 'handuk','tidak ada kotoran']
}

sentiment_lexicon = {
    'positive': ['bagus', 'baik', 'keren', 'puas', 'memuaskan', 'strategis', 'luas', 'lengkap', 'ramah', 'sopan', 'membantu', 'cepat', 'gercep', 'sigap', 'bersih', 'rapi', 'wangi', 'nyaman', 'enak', 'lezat', 'sempurna', 'luar biasa'],
    'negative': ['buruk', 'jelek', 'kecewa', 'kotor', 'bau', 'berdebu', 'berisik', 'lama', 'lambat', 'jutek', 'tidak ramah', 'rusak', 'aneh', 'mahal','enggak puas','kecewa'],
    'soft_negation': ['kurang'],
    'hard_negation': ['tidak'],
    'neutral': ['cukup', 'standar', 'biasa', 'lumayan', 'sesuai', 'saja', 'oke', 'agak']
}
print("Kamus kata kunci (versi final) untuk endpoint /predict_aspects berhasil dimuat.")
print("="*50)


# ==============================================================================
# ENDPOINT (ALAMAT URL)
# ==============================================================================

@app.route('/')
def home():
    return "<h1>API Analisis Sentimen Aktif!</h1><p>Gunakan endpoint /predict atau /predict_aspects.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if model and vectorizer:
        try:
            data = request.get_json(force=True)
            review_text = data['review_text']
            vectorized_text = vectorizer.transform([review_text])
            prediction = model.predict(vectorized_text)
            return jsonify({'prediction': prediction[0]})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Model untuk prediksi umum tidak berhasil dimuat.'}), 500


# ENDPOINT 2: Prediksi Skor per Aspek (LOGIKA DENGAN ATURAN BERTINGKAT)
@app.route('/predict_aspects', methods=['POST'])
def predict_aspects():
    try:
        data = request.get_json(force=True)
        review_text = data.get('review_text', '').lower()

        if not review_text:
            return jsonify({'error': "Key 'review_text' tidak ditemukan atau kosong."}), 400
        
        aspect_scores = {}
        
        for aspect, keywords in aspect_keywords.items():
            aspect_score = 0
            is_mentioned = False
            
            relevant_clauses = []
            clauses = re.split(r'[.!?]| tapi | namun | dan |,', review_text)
            for clause in clauses:
                if any(keyword in clause for keyword in keywords):
                    relevant_clauses.append(clause.strip())
                    is_mentioned = True

            if is_mentioned:
                for clause in relevant_clauses:
                    if any(f"tidak {pos_word}" in clause for pos_word in sentiment_lexicon['positive']):
                        aspect_score -= 1; continue
                    if any(f"kurang {pos_word}" in clause for pos_word in sentiment_lexicon['positive']):
                        aspect_score += 0; continue
                    if any(word in clause for word in sentiment_lexicon['neutral']):
                        aspect_score += 0; continue
                    has_positive = any(word in clause for word in sentiment_lexicon['positive'])
                    has_negative = any(word in clause for word in sentiment_lexicon['negative'])
                    if has_positive and has_negative:
                        aspect_score += 0; continue
                    if has_negative:
                        aspect_score -= 1; continue
                    if has_positive:
                        aspect_score += 1; continue
                aspect_scores[aspect] = aspect_score

        # ========================================================
        # PERUBAHAN DI SINI: HANYA PROSES ASPEK YANG DISEBUTKAN
        # ========================================================
        final_ratings = {}
        # Iterasi hanya pada aspek yang sudah dihitung skornya
        for aspect, score in aspect_scores.items():
            if score > 0:
                final_ratings[aspect] = 5.0
            elif score < 0:
                final_ratings[aspect] = 1.0
            else: # score == 0
                final_ratings[aspect] = 3.0
        # Tidak ada 'else' lagi, jadi aspek yang tidak disebut tidak akan masuk ke hasil

        return jsonify(final_ratings)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ==============================================================================
# Menjalankan Aplikasi
# ==============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)