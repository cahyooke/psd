import joblib
import streamlit as st
from hpelm import ELM  # penting! agar class dikenali sebelum load


# ==============================
# Load model dan encoder
# ==============================
elm = joblib.load("model_elm_heart.joblib")
encoder = joblib.load("encoder_elm_heart.joblib")

st.title("💖 Prediksi Penyakit Jantung dengan ELM")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko terkena penyakit jantung.")

# ==============================
# Input fitur dari user
# ==============================
age = st.number_input("Umur", min_value=1, max_value=120, value=40)
sex = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
cp = st.selectbox("Tipe Nyeri Dada (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan Darah (mmHg)", min_value=50, max_value=250, value=130)
chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Gula Darah > 120 mg/dl", [0, 1])
restecg = st.selectbox("Hasil ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=250, value=150)
exang = st.selectbox("Angina karena olahraga", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope ST", [0, 1, 2])
ca = st.selectbox("Jumlah pembuluh besar (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# ==============================
# Prediksi
# ==============================
if st.button("Prediksi"):
    input_data = np.array([[age, 1 if sex == "Laki-laki" else 0, cp, trestbps, chol,
                            fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    pred = elm.predict(input_data)
    hasil = encoder.categories_[0][pred][0]

    st.subheader("🩺 Hasil Prediksi:")
    if hasil == "1":
        st.error("⚠️ Pasien berisiko terkena penyakit jantung.")
    else:
        st.success("✅ Pasien tidak berisiko penyakit jantung.")
