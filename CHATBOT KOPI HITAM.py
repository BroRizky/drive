# Import library yang dibutuhkan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Database Pengetahuan (Corpus) ---
# Pasangan [Pertanyaan, Jawaban]
knowledge_base = [
    ["Halo", "Halo! Ada yang bisa saya bantu?"],
    ["Apa kabar?", "Kabar saya baik, terima kasih!"],
    ["Siapa namamu?", "Saya adalah chatbot AI Kopihitam."],
    ["Kamu bisa apa?", "Saya bisa menjawab pertanyaan berdasarkan database saya."],
    ["Terima kasih", "Sama-sama! Senang bisa membantu."],
    ["Selamat pagi", "Selamat pagi! Semoga hari Anda menyenangkan."],
    ["Bagaimana cuaca hari ini?", "Maaf, saya tidak terhubung ke data cuaca saat ini."],
    ["bye", "Sampai jumpa!"]
]

# Pisahkan pertanyaan dan jawaban
# .lower() untuk membuat semuanya huruf kecil
questions = [item[0].lower() for item in knowledge_base]
answers = [item[1] for item in knowledge_base]

# --- 2. Proses "Belajar" (Vectorizing) ---
# Ubah teks pertanyaan menjadi vektor angka (TF-IDF)
vectorizer = TfidfVectorizer()
X_questions_vec = vectorizer.fit_transform(questions) # "Melatih" vectorizer

# --- 3. Fungsi untuk Mendapatkan Respon ---
def get_bot_response(user_input):
    """
    Mencari jawaban yang paling mirip dari knowledge_base.
    """
    # 1. Ubah input pengguna menjadi vektor
    user_vec = vectorizer.transform([user_input.lower()])
    
    # 2. Hitung kesamaan (cosine similarity) antara input pengguna dan semua pertanyaan
    similarities = cosine_similarity(user_vec, X_questions_vec)
    
    # 3. Dapatkan skor kesamaan tertinggi
    max_similarity = np.max(similarities)
    
    # 4. Tentukan batas minimum (threshold)
    # Jika tidak ada yang cukup mirip (misal, di bawah 20%), anggap tidak mengerti
    if max_similarity < 0.2:
        return "Maaf, saya tidak mengerti apa yang Anda maksud."
    else:
        # 5. Ambil indeks dari pertanyaan yang paling mirip
        most_similar_index = np.argmax(similarities)
        return answers[most_similar_index]

# --- 4. Loop Percakapan Utama ---
print("Halo! Saya chatbot AI (Kopi Hitam). Ketik 'exit' untuk keluar.")
while True:
    user_text = input("Anda: ")
    if user_text.lower() == 'exit' or user_text.lower() == 'keluar':
        print("Bot: Sampai jumpa!")
        break
    
    # Dapatkan respon dari fungsi
    response = get_bot_response(user_text)
    print(f"Bot: {response}")