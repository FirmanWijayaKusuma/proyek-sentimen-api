import joblib
import re
from flask import Flask, request, jsonify
from pathlib import Path
from flask_cors import CORS # 

# ==============================================================================
# Inisialisasi Aplikasi Flask
# ==============================================================================
app = Flask(__name__)
CORS(app) # <-- 2. TERAPKAN CORS KE APLIKASI ANDA

# --- Menentukan Path Absolut ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model_svc.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"

# ... (Sisa seluruh kode Anda dari sini ke bawah tetap SAMA, tidak ada yang perlu diubah) ...

# --- Memuat Model dan Kamus ---
try:
    print(f"Mencoba memuat model dari: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("="*50)
    print("Model ML dan Vectorizer berhasil dimuat.")
except Exception as e:
    print(f"Peringatan: Gagal memuat model. Error: {e}")
    model = None
    vectorizer = None

aspect_keywords = {
    'Fasilitas': ['fasilitas', 'lokasi', 'kamar', 'kamar mandi', 'ac', 'tv', 'wifi', 'kolam renang', 'parkir', 'transportasi', 'restoran', 'makanan', 'menu', 'sarapan', 'lengkap'],
    'Staf': ['staff', 'staf', 'pelayanan', 'karyawan', 'resepsionis', 'petugas', 'ramah', 'sopan', 'membantu', 'gercep', 'sigap', 'respon', 'lambat', 'jutek','layanan'],
    'Kebersihan': ['bersih', 'kebersihan', 'rapi', 'wangi', 'nyaman', 'kotor', 'bau', 'debu', 'sprei', 'handuk','tidak ada kotoran']
}
sentiment_lexicon = {
    'positive': ['bagus', 'baik', 'keren', 'puas', 'memuaskan', 'strategis', 'luas', 'lengkap', 'ramah', 'sopan', 'membantu', 'cepat', 'gercep', 'sigap', 'bersih', 'rapi', 'wangi', 'nyaman', 'enak', 'lezat', 'sempurna', 'luar biasa'],
    'negative': ['buruk', 'jelek', 'kecewa', 'kotor', 'bau', 'berdebu', 'berisik', 'lama', 'lambat', 'jutek', 'tidak ramah', 'rusak', 'aneh', 'mahal','enggak puas'],
    'soft_negation': ['kurang'],
    'hard_negation': ['tidak'],
    'neutral': ['cukup', 'standar', 'biasa', 'lumayan', 'sesuai', 'saja', 'oke', 'agak']
}
print("Kamus kata kunci berhasil dimuat.")
print("="*50)

@app.route('/')
def home():
    return "<h1>API Analisis Sentimen Aktif!</h1>"

@app.route('/predict_aspects', methods=['POST'])
def predict_aspects():
    try:
        data = request.get_json(force=True)
        review_text = data.get('review_text', '').lower()
        if not review_text: return jsonify({'error': "Key 'review_text' tidak ditemukan atau kosong."}), 400
        
        final_ratings = {}
        for aspect, keywords in aspect_keywords.items():
            aspect_score = 0
            is_mentioned = False
            relevant_clauses = [c.strip() for c in re.split(r'[.!?]| tapi | namun | dan |,', review_text) if any(kw in c for kw in keywords)]

            if relevant_clauses:
                is_mentioned = True
                for clause in relevant_clauses:
                    if any(f"tidak {pos}" in clause for pos in sentiment_lexicon['positive']): aspect_score -= 1; continue
                    if any(f"kurang {pos}" in clause for pos in sentiment_lexicon['positive']): aspect_score += 0; continue
                    if any(neu in clause for neu in sentiment_lexicon['neutral']): aspect_score += 0; continue
                    has_pos = any(pos in clause for pos in sentiment_lexicon['positive']); has_neg = any(neg in clause for neg in sentiment_lexicon['negative'])
                    if has_pos and has_neg: aspect_score += 0
                    elif has_neg: aspect_score -= 1
                    elif has_pos: aspect_score += 1
                
            if is_mentioned:
                if aspect_score > 0: final_ratings[aspect] = 5.0
                elif aspect_score < 0: final_ratings[aspect] = 1.0
                else: final_ratings[aspect] = 3.0
        
        return jsonify(final_ratings)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)