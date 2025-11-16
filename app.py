# app.py (Versi Hosting - Tanpa Scraper)
import streamlit as st
import pandas as pd
import time
import re
import string
import os 
import base64 

# --- Impor Selenium DIHAPUS ---

# --- Impor untuk ML & Plotly ---
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

# --- Komponen dari Kode 2: Stemmer (Opsional) ---
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except Exception:
    stemmer = None

# --- Komponen dari Kode 2: Stopwords & Lexicon ---
INDO_STOPWORDS = {
    'yang','dan','di','ke','dari','dengan','pada','atau','itu',' untuk','untuk','akan','saya','aku',
    'kamu','anda','dia','kami','kita','tidak','tdk','ga','gak','yg','tapi','karena','sebagai','adalah',
    'apa','siapa','ini','itu','dari','ke','sebagai','oleh','lagi','sudah','belum'
}
POSITIVE_WORDS = {
    'bagus','baik','mantap','terbaik','lancar','membantu','mudah','cepat','inspiratif',
    'hebat','positif','berhasil','menarik','keren','puas','suka','rekomendasi','support'
}
NEGATIVE_WORDS = {
    'buruk','jelek','lambat','sulit','error','gagal','negatif','tidak','kurang','benci',
    'menyusahkan','parah','lemah','ribet'
}
NEGATION_PHRASES = {
    'tidak bagus': -2, 'tidak baik': -2, 'tidak membantu': -2, 'tidak akurat': -2,
    'tidak efektif': -2, 'kurang bagus': -2, 'kurang baik': -2, 'kurang akurat': -2,
    'kurang membantu': -2,
}

# --- Fungsi Scraper (get_tweet, scrape_tweets) DIHAPUS ---

# ------------------ Fungsi Preprocessing ------------------
def clean_text(text: str) -> str:
    if pd.isna(text): return ''
    txt = str(text).lower()
    txt = re.sub(r'http\S+|www\.\S+', ' ', txt)
    txt = re.sub(r'@\w+|#\w+', ' ', txt)
    txt = re.sub(r'[^a-z\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    tokens = [w for w in txt.split() if w not in INDO_STOPWORDS]
    if stemmer:
        try: tokens = [stemmer.stem(w) for w in tokens]
        except Exception: pass 
    return ' '.join(tokens)

# ------------------ Fungsi Labeling ------------------
def lexicon_sentiment(text: str) -> str:
    if not text: return 'netral'
    txt = text.lower() 
    score = 0
    for phrase, val in NEGATION_PHRASES.items():
        if phrase in txt:
            score += val
            txt = txt.replace(phrase, '') 
    for w in txt.split():
        if w in ['tidak','kurang']: continue
        if w in POSITIVE_WORDS: score += 1
        elif w in NEGATIVE_WORDS: score -= 1
    if score > 0: return 'positif'
    elif score < 0: return 'negatif'
    return 'netral'

# ------------------ Fungsi Training & Evaluasi ------------------
def train_and_evaluate(df: pd.DataFrame, text_col='clean_text', label_col='sentiment'):
    if label_col not in df.columns:
        raise ValueError(f"Kolom label '{label_col}' tidak ditemukan.")
    df_clean = df[[text_col, label_col]].dropna(subset=[text_col, label_col])
    df_clean = df_clean[df_clean[text_col].str.strip() != ''] 
    if df_clean.shape[0] < 10:
        raise ValueError("Data terlalu sedikit untuk training (butuh minimal 10 baris).")
    if df_clean[label_col].nunique() < 2:
        raise ValueError(f"Hanya ditemukan 1 kelas sentimen: '{df_clean[label_col].unique()}'. Tidak bisa melakukan train/test split.")
    X = df_clean[text_col].astype(str)
    y = df_clean[label_col].astype(str)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        st.warning("Tidak bisa melakukan stratify split. Menggunakan split biasa.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_f1 = f1_score(y_test, nb_pred, average='weighted')
    nb_report = classification_report(y_test, nb_pred, output_dict=False)
    svm = LinearSVC(max_iter=10000, class_weight='balanced')
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')
    svm_report = classification_report(y_test, svm_pred, output_dict=False)
    results = {
        'vectorizer': vectorizer, 'nb_model': nb, 'svm_model': svm,
        'nb_f1': nb_f1, 'svm_f1': svm_f1, 'nb_report': nb_report, 'svm_report': svm_report,
        'test_df': pd.DataFrame({'text': X_test, 'actual': y_test, 'nb_pred': nb_pred, 'svm_pred': svm_pred}).reset_index(drop=True)
    }
    return results

# ==================================================================
#                     STREAMLIT APP UTAMA
# ==================================================================

st.set_page_config(page_title="Analisis Sentimen ChatAI", layout="wide")

# --- Sidebar ---
if os.path.exists("untar_logo.png"):
    st.sidebar.image("untar_logo.png", width=150)
else:
    st.sidebar.warning("File `untar_logo.png` tidak ditemukan.")

st.sidebar.title("Navigasi")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- Halaman Login ---
if not st.session_state["logged_in"]:
    st.title("🔐 Login Sistem Analisis Sentimen")
    st.markdown("Silakan login untuk menggunakan aplikasi.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login berhasil!")
            time.sleep(1)
            st.rerun() 
        else:
            st.error("Username atau password salah.")
    st.stop()


# --- Tampilan Setelah Login ---
st.sidebar.success("Anda sudah login.")
# --- "Scrape Data" TETAP ADA di navigasi ---
page = st.sidebar.radio("Pilih halaman:", ["Dashboard", "Scrape Data", "Upload & Proses Data", "Visualisasi Hasil"])

# --- Halaman Dashboard ---
if page == "Dashboard":
    st.title("Perancangan Sistem Informasi untuk Menganalisis Sentimen Pengguna terhadap ChatAI dalam Dunia Pendidikan")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👨‍🎓 Peneliti")
        st.markdown("**ARMANTA TARIGAN** (825220129)")
    with col2:
        st.subheader("👨‍🏫 Pembimbing")
        st.markdown("**DEDI TRISNAWARMAN, S.Si., M.Kom., Dr.**")
    
    st.markdown("---")

    st.header("Ringkasan Penelitian")
    st.info("""
    Penelitian ini bertujuan untuk menganalisis sentimen pengguna terhadap penggunaan ChatAI
    dalam dunia pendidikan, khususnya di kalangan mahasiswa Indonesia, dengan menggunakan
    metode machine learning seperti Naive Bayes dan Support Vector Machine (SVM). 
    Data diperoleh melalui kuesioner dan platform media sosial (Twitter/Reddit), yang kemudian diproses
    melalui teknik Natural Language Processing (NLP). Model dievaluasi berdasarkan F1-score 
    untuk memilih model terbaik dan hasilnya disajikan dalam dashboard visual interaktif ini.
    """)
    
    st.header("Alasan Pengambilan Topik (Latar Belakang)")
    st.markdown("""
    Alasan utama di balik penelitian ini adalah maraknya penggunaan ChatAI (seperti ChatGPT, Gemini, atau Copilot) oleh mahasiswa sebagai asisten digital untuk membantu menyusun ide, menjawab pertanyaan, hingga mengerjakan tugas kuliah.
    
    Namun, penggunaan ini menimbulkan persepsi yang beragam di kalangan mahasiswa. Sebagian mahasiswa menganggap ChatAI sebagai alat bantu yang efektif dan efisien, sementara sebagian lainnya khawatir penggunaannya dapat menurunkan kemampuan berpikir kritis dan orisinalitas.
    
    **Oleh karena itu, diperlukan analisis yang lebih mendalam** untuk memahami bagaimana sentimen dan persepsi mahasiswa Indonesia terhadap penggunaan ChatAI sebagai alat bantu akademik. Penelitian ini diharapkan dapat memberikan gambaran yang jelas mengenai pandangan tersebut.
    """)
    
    st.header("Metodologi dan Alur Sistem")
    st.markdown("""
    Aplikasi ini dirancang untuk mengikuti alur metodologi penelitian yang telah didefinisikan.
    Proses dimulai dari pengumpulan data, preprocessing, hingga evaluasi model dan visualisasi.
    """)

    tab_metodologi, tab_alur = st.tabs(["Bagan Metodologi Penelitian (Gambar 1)", "Diagram Alur Sistem (Gambar 2)"])

    with tab_metodologi:
        st.subheader("Bagan Tahapan Metodologi Penelitian")
        if os.path.exists("gambar_1_metodologi.png"):
            st.image("gambar_1_metodologi.png", caption="Sumber: Proposal Skripsi (Gambar 1)")
        else:
            st.error("File `gambar_1_metodologi.png` tidak ditemukan.")
        
        st.markdown("""
        [cite_start]Bagan ini menunjukkan 4 langkah utama penelitian [cite: 121-124]:
        1.  **Pengumpulan Data:** Mengambil data dari Twitter dan Kuesioner. (Tahap scraping data Twitter dilakukan secara lokal).
        2.  **Preprocessing:** Membersihkan data (cleaning, tokenizing, stemming) dan memberi label (lexicon).
        3.  **Penerapan Algoritma:** Menggunakan TF-IDF untuk ekstraksi fitur dan melatih model Naive Bayes serta SVM.
        4.  **Evaluasi dan Analisis:** Menghitung F1-Score dan memvisualisasikan hasil.
        """)

    with tab_alur:
        st.subheader("Diagram Alur Sistem Analisis Sentimen")
        if os.path.exists("gambar_2_alur_sistem.png"):
            st.image("gambar_2_alur_sistem.png", caption="Sumber: Proposal Skripsi (Gambar 2)")
        else:
            st.error("File `gambar_2_alur_sistem.png` tidak ditemukan.")
        
        st.markdown("""
        [cite_start]Diagram ini mendetailkan proses yang terjadi di dalam aplikasi ini [cite: 308-336]:
        * **Input Data:** Anda mengunggah file CSV/XLSX dari Twitter dan Kuesioner.
        * **Preprocessing Data:** Teks dibersihkan (case folding, cleansing, tokenization, stopword, stemming).
        * **Labeling Sentimen:** Teks yang bersih diberi label (positif, negatif, netral) menggunakan kamus lexicon.
        * **Ekstraksi Fitur TF-IDF:** Teks diubah menjadi angka (vektor) agar bisa dibaca oleh machine learning.
        * **Pemodelan Machine Learning:** Data latih digunakan untuk melatih model Naive Bayes (NB) dan Support Vector Machine (SVM).
        * **Evaluasi Model:** Kinerja model diukur menggunakan Accuracy, Precision, Recall, dan F1-Score.
        * **Visualisasi:** Hasilnya ditampilkan dalam bentuk grafik dan tabel di halaman **'Visualisasi Hasil'**.
        """)
    
    st.markdown("---")
    st.header("🖼️ Tampilan Sistem (Mockup)")
    st.info("Berikut adalah tampilan fitur utama dari sistem yang dirancang, mulai dari upload data mentah hingga visualisasi hasil analisis sentimen.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Upload Data")
        if os.path.exists("image_eabe02.jpg"): 
            st.image("image_eabe02.jpg", caption="Halaman 'Upload & Proses Data'")
        else:
            st.warning("File 'image_eabe02.jpg' tidak ditemukan.")
        st.markdown("Unggah data mentah (Twitter/Kuesioner) dan pilih kolom teks yang akan dianalisis.")

    with col2:
        st.subheader("2. Evaluasi Model")
        if os.path.exists("image_ea6714.jpg"): 
            st.image("image_ea6714.jpg", caption="Perbandingan F1-Score di Halaman Visualisasi")
        else:
            st.warning("File 'image_ea6714.jpg' tidak ditemukan.")
        st.markdown("Sistem otomatis melatih Naive Bayes & SVM, lalu menampilkan perbandingan F1-Score.")

    with col3:
        st.subheader("3. Visualisasi Hasil")
        if os.path.exists("image_eabe00.png"): 
            st.image("image_eabe00.png", caption="Grafik Distribusi Sentimen")
        else:
            st.warning("File 'image_eabe00.png' tidak ditemukan.")
        st.markdown("Lihat perbandingan sentimen Twitter vs Kuesioner dalam bentuk Pie Chart dan Bar Chart.")


# --- HALAMAN SCRAPE DATA (DIGANTI DENGAN PESAN) ---
elif page == "Scrape Data":
    st.header("🕸️ Scrape Data dari X (Twitter)")
    st.warning("🔒 Fitur Scraping Dinonaktifkan di Versi Hosting")
    st.markdown("""
    Fitur *scraping* data secara langsung dari X (Twitter) menggunakan Selenium **tidak dapat dijalankan di server hosting publik** seperti Streamlit Community Cloud. 
    
    Fitur ini membutuhkan instalasi *driver* (GeckoDriver) dan *browser* (Firefox) di tingkat sistem operasi, yang tidak diizinkan di lingkungan server bersama.
    
    ### 👨‍💻 Alur Kerja untuk Demo/Presentasi:
    1.  **Jalankan di Lokal:** Gunakan file Python versi lengkap (yang ada Selenium) di komputer lokal Anda untuk menjalankan fitur *scraping* ini.
    2.  **Unduh Hasil:** Simpan hasil *scraping* sebagai file `.csv` (misal: `tweets_scraped.csv`).
    3.  **Unggah ke Server:** Kembali ke aplikasi *hosting* ini, buka halaman **"Upload & Proses Data"**, dan unggah file `.csv` tersebut di tab "Proses Data Twitter".
    """)
    
    st.info("Halaman ini tetap ditampilkan untuk menunjukkan bahwa fitur *scraping* adalah bagian dari perancangan sistem, sesuai dengan **Tahap 1: Pengumpulan Data**.")


# --- HALAMAN UPLOAD & PROSES DATA ---
elif page == "Upload & Proses Data":
    st.header("📤 Upload & Proses Data")
    st.info("""
    Halaman ini adalah inti dari sistem, mencakup **Tahap 2 (Preprocessing & Labeling)** dan **Tahap 3 (Penerapan Algoritma)**.
    Silakan unggah data Anda pada tab yang sesuai.
    """)

    tab1, tab2 = st.tabs(["Proses Data Twitter", "Proses Data Kuesioner"])

    with tab1:
        st.subheader("Proses Data Twitter (Data Sekunder)")
        uploaded_twitter = st.file_uploader("Unggah data Twitter (CSV/XLSX)", type=["csv", "xlsx"], key="twitter_upload")
        
        if uploaded_twitter:
            try:
                df_twitter_raw = pd.read_csv(uploaded_twitter) if uploaded_twitter.name.endswith(".csv") else pd.read_excel(uploaded_twitter)
                st.session_state.df_twitter_raw = df_twitter_raw
                st.success(f"✅ Data Twitter '{uploaded_twitter.name}' berhasil diunggah!")
                st.dataframe(df_twitter_raw.head())
            except Exception as e:
                st.error(f"Gagal membaca file Twitter: {e}")

        if "df_twitter_raw" in st.session_state:
            df_tw_raw = st.session_state.df_twitter_raw.copy() 
            if 'Text' not in df_tw_raw.columns:
                st.error(f"Kolom 'Text' tidak ditemukan di file Twitter. Kolom yang ada: {list(df_tw_raw.columns)}")
            else:
                if st.button("Mulai Proses Sentimen Twitter", key="proses_twitter"):
                    try:
                        with st.spinner("🔄 Sedang preprocessing (Twitter)..."):
                            df_tw_raw['clean_text'] = df_tw_raw['Text'].astype(str).apply(clean_text)
                        with st.spinner("🔄 Sedang labeling (Twitter)..."):
                            df_tw_raw['sentiment'] = df_tw_raw['clean_text'].apply(lexicon_sentiment)
                        st.session_state['df_twitter_processed'] = df_tw_raw
                        st.success("✅ Preprocessing & Labeling Twitter selesai.")
                        
                        with st.spinner("🧠 Melatih model (Twitter)..."):
                            results = train_and_evaluate(df_tw_raw, text_col='clean_text', label_col='sentiment')
                            st.session_state['twitter_model_results'] = results
                        st.success("✅ Training & evaluasi Twitter selesai.")
                        
                        st.subheader("Hasil F1-Score (Twitter)")
                        col_nb, col_svm = st.columns(2)
                        col_nb.metric("F1-Score Naive Bayes", f"{results['nb_f1']:.3f}")
                        col_svm.metric("F1-Score SVM", f"{results['svm_f1']:.3f}")
                        
                        st.subheader("Classification Report (Twitter)")
                        col_rep1, col_rep2 = st.columns(2)
                        col_rep1.text("=== Naive Bayes ===")
                        col_rep1.text(results['nb_report'])
                        col_rep2.text("=== SVM ===")
                        col_rep2.text(results['svm_report'])

                        st.markdown("---")
                        st.subheader("⬇️ Preview & Unduh Hasil (Before & After)")
                        
                        col_dw1, col_dw2 = st.columns(2)
                        with col_dw1:
                            st.markdown("**Data ASLI (Before)**")
                            st.dataframe(st.session_state['df_twitter_raw'].head(5), use_container_width=True)
                            csv_before = st.session_state['df_twitter_raw'].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Unduh Data ASLI (Before)",
                                data=csv_before,
                                file_name="twitter_data_ASLI.csv",
                                mime="text/csv",
                                key="btn_tw_before"
                            )
                        with col_dw2:
                            st.markdown("**Data PROSES (After)**")
                            st.dataframe(df_tw_raw.head(5), use_container_width=True)
                            csv_after = df_tw_raw.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Unduh Data PROSES (After)",
                                data=csv_after,
                                file_name="twitter_data_PROSES.csv",
                                mime="text/csv",
                                key="btn_tw_after"
                            )

                    except Exception as e:
                        st.error(f"Gagal saat proses Twitter: {e}")

    with tab2:
        st.subheader("Proses Data Kuesioner (Data Primer)")
        st.markdown("Data ini adalah **Data Primer** yang menyediakan opini terstruktur berdasarkan pertanyaan riset.")
        uploaded_kuesioner = st.file_uploader("Unggah data Kuesioner (CSV/XLSX)", type=["csv", "xlsx"], key="kues_upload")
        
        text_col_kues = None
        
        if uploaded_kuesioner:
            try:
                df_kues_raw = pd.read_csv(uploaded_kuesioner) if uploaded_kuesioner.name.endswith(".csv") else pd.read_excel(uploaded_kuesioner)
                st.session_state.df_kues_raw = df_kues_raw
                st.success(f"✅ Data Kuesioner '{uploaded_kuesioner.name}' berhasil diunggah!")
                st.dataframe(df_kues_raw.head())
                
                likely_cols = [col for col in df_kues_raw.columns if df_kues_raw[col].dtype == 'object']
                likely_cols = [col for col in likely_cols if 'timestamp' not in col.lower() and 'nama' not in col.lower()]
                default_index = 0
                if likely_cols:
                    default_index = list(df_kues_raw.columns).index(likely_cols[-1]) 
                
                text_col_kues = st.selectbox(
                    "Pilih kolom yang berisi Teks/Feedback/Saran:",
                    options=df_kues_raw.columns,
                    index=default_index,
                    key="select_kues_col"
                )
                
            except Exception as e:
                st.error(f"Gagal membaca file Kuesioner: {e}")

        if "df_kues_raw" in st.session_state and text_col_kues:
            if st.button("Mulai Proses Sentimen Kuesioner", key="proses_kues"):
                try:
                    df_k_raw = st.session_state.df_kues_raw.copy() 
                    with st.spinner("🔄 Sedang preprocessing (Kuesioner)..."):
                        df_k_raw['clean_text'] = df_k_raw[text_col_kues].astype(str).apply(clean_text)
                    with st.spinner("🔄 Sedang labeling (Kuesioner)..."):
                        df_k_raw['sentiment'] = df_k_raw['clean_text'].apply(lexicon_sentiment)
                    st.session_state['df_kues_processed'] = df_k_raw
                    st.success("✅ Preprocessing & Labeling Kuesioner selesai.")
                    
                    st.dataframe(df_k_raw[[text_col_kues, 'clean_text', 'sentiment']].head())
                    
                    with st.spinner("🧠 Melatih model (Kuesioner)..."):
                        results = train_and_evaluate(df_k_raw, text_col='clean_text', label_col='sentiment')
                        st.session_state['kues_model_results'] = results
                    st.success("✅ Training & evaluasi Kuesioner selesai.")
                    
                    st.subheader("Hasil F1-Score (Kuesioner)")
                    col_nb_k, col_svm_k = st.columns(2)
                    col_nb_k.metric("F1-Score Naive Bayes", f"{results['nb_f1']:.3f}")
                    col_svm_k.metric("F1-Score SVM", f"{results['svm_f1']:.3f}")

                    st.subheader("Classification Report (Kuesioner)")
                    col_repk1, col_repk2 = st.columns(2)
                    col_repk1.text("=== Naive Bayes ===")
                    col_repk1.text(results['nb_report'])
                    col_repk2.text("=== SVM ===")
                    col_repk2.text(results['svm_report'])

                    st.markdown("---")
                    st.subheader("⬇️ Preview & Unduh Hasil (Before & After)")
                    
                    col_dw_k1, col_dw_k2 = st.columns(2)
                    with col_dw_k1:
                        st.markdown("**Data ASLI (Before)**")
                        st.dataframe(st.session_state['df_kues_raw'].head(5), use_container_width=True)
                        csv_before_kues = st.session_state['df_kues_raw'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Data ASLI (Before)",
                            data=csv_before_kues,
                            file_name="kuesioner_data_ASLI.csv",
                            mime="text/csv",
                            key="btn_kues_before"
                        )
                    with col_dw_k2:
                        st.markdown("**Data PROSES (After)**")
                        st.dataframe(df_k_raw.head(5), use_container_width=True)
                        csv_after_kues = df_k_raw.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Data PROSES (After)",
                            data=csv_after_kues,
                            file_name="kuesioner_data_PROSES.csv",
                            mime="text/csv",
                            key="btn_kues_after"
                        )
                
                except Exception as e:
                    st.error(f"Gagal saat proses Kuesioner: {e}")

# --- HALAMAN VISUALISASI ---
elif page == "Visualisasi Hasil":
    st.header("📊 Visualisasi Hasil Analisis Sentimen")
    st.info("""
    Halaman ini adalah implementasi dari **Tahap 4: Evaluasi dan Analisis**.
    Di sini, kita membandingkan F1-Score model dan melihat distribusi sentimen 
    keseluruhan dari data yang telah diproses.
    """)

    has_twitter = 'twitter_model_results' in st.session_state
    has_kues = 'kues_model_results' in st.session_state

    if not has_twitter and not has_kues:
        st.warning("Belum ada data yang diproses. Silakan jalankan proses di 'Upload & Proses Data'.")
        st.stop()

    st.subheader("🔹 Perbandingan F1-Score Model")
    col1, col2 = st.columns(2)
    with col1:
        if has_twitter:
            st.markdown("#### F1-Score (Data Twitter)")
            res_tw = st.session_state['twitter_model_results']
            data_tw = pd.DataFrame({'Model': ['Naive Bayes', 'SVM'], 'F1 Score': [res_tw['nb_f1'], res_tw['svm_f1']]})
            fig_f1_tw = px.bar(data_tw, x='Model', y='F1 Score', color='Model', text='F1 Score', range_y=[0,1], 
                               color_discrete_sequence=['#00C2A8', '#A66CFF'])
            fig_f1_tw.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_f1_tw.update_layout(template='plotly_white')
            st.plotly_chart(fig_f1_tw, use_container_width=True)
            best_tw = data_tw.loc[data_tw['F1 Score'].idxmax(), 'Model']
            st.success(f"🏆 Model Twitter terbaik: **{best_tw}**")
        else:
            st.info("Data Twitter belum diproses.")
            
    with col2:
        if has_kues:
            st.markdown("#### F1-Score (Data Kuesioner)")
            res_ku = st.session_state['kues_model_results']
            data_ku = pd.DataFrame({'Model': ['Naive Bayes', 'SVM'], 'F1 Score': [res_ku['nb_f1'], res_ku['svm_f1']]})
            fig_f1_ku = px.bar(data_ku, x='Model', y='F1 Score', color='Model', text='F1 Score', range_y=[0,1], 
                               color_discrete_sequence=['#FF6B6B', '#FFD166'])
            fig_f1_ku.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_f1_ku.update_layout(template='plotly_white')
            st.plotly_chart(fig_f1_ku, use_container_width=True)
            best_ku = data_ku.loc[data_ku['F1 Score'].idxmax(), 'Model']
            st.success(f"🏆 Model Kuesioner terbaik: **{best_ku}**")
        else:
            st.info("Data Kuesioner belum diproses.")

    st.divider()
    st.subheader("🔹 Distribusi Sentimen (Hasil Labeling)")
    col_pie1, col_pie2 = st.columns(2)
    
    with col_pie1:
        if 'df_twitter_processed' in st.session_state:
            st.markdown("#### 🍥 Distribusi Sentimen Twitter")
            dfp_tw = st.session_state['df_twitter_processed']
            fig_twitter_pie = px.pie(dfp_tw, names='sentiment', color='sentiment',
                                     color_discrete_sequence=px.colors.qualitative.Safe,
                                     title="Twitter Sentiment Distribution")
            fig_twitter_pie.update_layout(template='plotly_white')
            st.plotly_chart(fig_twitter_pie, use_container_width=True)
        else:
            st.info("Data Twitter belum diproses.")
            
    with col_pie2:
        if 'df_kues_processed' in st.session_state:
            st.markdown("#### 🍥 Distribusi Sentimen Kuesioner")
            dfp_ku = st.session_state['df_kues_processed']
            fig_kues_pie = px.pie(dfp_ku, names='sentiment', color='sentiment',
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  title="Kuesioner Sentiment Distribution")
            fig_kues_pie.update_layout(template='plotly_white')
            st.plotly_chart(fig_kues_pie, use_container_width=True)
        else:
            st.info("Data Kuesioner belum diproses.")

    st.divider()
    st.subheader("🔹 Perbandingan Distribusi Sentimen: Twitter vs Kuesioner")
    
    if 'df_twitter_processed' in st.session_state and 'df_kues_processed' in st.session_state:
        df_twitter = st.session_state['df_twitter_processed']
        df_kues = st.session_state['df_kues_processed']
        twitter_count = df_twitter['sentiment'].value_counts()
        kues_count = df_kues['sentiment'].value_counts()
        df_compare = pd.concat([
            twitter_count.rename("Twitter"),
            kues_count.rename("Kuesioner")
        ], axis=1).fillna(0).reset_index()
        df_compare = df_compare.rename(columns={df_compare.columns[0]: 'Sentiment'})
        df_compare['Twitter (%)'] = (df_compare['Twitter'] / df_compare['Twitter'].sum() * 100).round(1)
        df_compare['Kuesioner (%)'] = (df_compare['Kuesioner'] / df_compare['Kuesioner'].sum() * 100).round(1)
        
        fig_compare = go.Figure(data=[
            go.Bar(name='Twitter', x=df_compare['Sentiment'], y=df_compare['Twitter'], marker_color='#00C2A8'),
            go.Bar(name='Kuesioner', x=df_compare['Sentiment'], y=df_compare['Kuesioner'], marker_color='#A66CFF')
        ])
        fig_compare.update_layout(
            barmode='group', template='plotly_white',
            title="Distribusi Sentimen: Twitter vs Kuesioner",
            xaxis_title="Sentiment", yaxis_title="Jumlah", legend_title="Sumber Data"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        st.markdown("### 📋 Tabel Perbandingan Persentase Sentimen")
        st.dataframe(df_compare[['Sentiment', 'Twitter (%)', 'Kuesioner (%)', 'Twitter', 'Kuesioner']], use_container_width=True)
        
    else:
        st.warning("⚠️ Untuk melihat perbandingan, Anda harus memproses data Twitter dan data Kuesioner di tab 'Upload & Proses Data'.")

# --- AKHIR DARI FILE ---
