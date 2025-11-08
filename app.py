import streamlit as st
import numpy as np
import joblib

# ============================================================
# 1. Definisi ulang fungsi aktivasi dan class ELMClassifier
# ============================================================
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

class ELMClassifier:
    def __init__(self, input_size, hidden_size, activation=sigmoid_activation, random_state=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.input_weights = np.random.randn(self.input_size, self.hidden_size)
        self.bias = np.random.randn(self.hidden_size)

    def fit(self, X, y):
        H = self.activation(np.dot(X, self.input_weights) + self.bias)
        self.output_weights = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        H = self.activation(np.dot(X, self.input_weights) + self.bias)
        output = np.dot(H, self.output_weights)
        return np.argmax(output, axis=1)

# ============================================================
# 2. Load model dan encoder
# ============================================================
try:
    model = joblib.load("model_elm_heart.joblib")
    encoder = joblib.load("encoder_elm_heart.joblib")
    scaler = joblib.load("scaler_elm_heart.joblib")  # kalau kamu punya scaler simpan juga
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# ============================================================
# 3. UI Streamlit
# ============================================================
st.title("💖 Prediksi Penyakit Jantung dengan ELM")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko terkena penyakit jantung.")

umur = st.number_input("Umur", 20, 100, 50)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
tipe_nyeri_dada = st.number_input("Tipe Nyeri Dada (0–3)", 0, 3, 0)
tekanan_darah = st.number_input("Tekanan Darah (mmHg)", 80, 200, 120)
kolesterol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
gula_darah = st.number_input("Gula Darah > 120 mg/dl (1=Ya,0=Tidak)", 0, 1, 0)
hasil_ecg = st.number_input("Hasil ECG (0–2)", 0, 2, 0)
denyut_jantung = st.number_input("Denyut Jantung Maksimum", 60, 210, 150)
angina = st.number_input("Angina karena olahraga (1=Ya,0=Tidak)", 0, 1, 0)
st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope_st = st.number_input("Slope ST (0–2)", 0, 2, 1)
pembuluh = st.number_input("Jumlah pembuluh besar (0–4)", 0, 4, 0)
thal = st.number_input("Thalassemia (0–3)", 0, 3, 1)

# ============================================================
# 4. Prediksi
# ============================================================
if st.button("🩺 Prediksi"):
    try:
        # Konversi jenis kelamin ke angka
        jk = 1 if jenis_kelamin == "Laki-laki" else 0

        data_input = np.array([[umur, jk, tipe_nyeri_dada, tekanan_darah, kolesterol,
                                gula_darah, hasil_ecg, denyut_jantung, angina,
                                st_depression, slope_st, pembuluh, thal]])

        # Normalisasi jika scaler tersedia
        if 'scaler' in locals():
            data_input_scaled = scaler.transform(data_input)
        else:
            data_input_scaled = data_input

        # Prediksi
        pred = model.predict(data_input_scaled)
        hasil = encoder.categories_[0][pred][0]

        if hasil == 1 or hasil == "1":
            st.error("⚠️ Pasien berisiko terkena penyakit jantung!")
        else:
            st.success("✅ Pasien tidak berisiko penyakit jantung.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
