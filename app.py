import streamlit as st
import joblib
import numpy as np
from elm_model import ELMClassifier, sigmoid_activation  # pastikan ini sama dengan nama class di file elm_model.py

# Load model dan encoder
model = joblib.load("model_elm_heart.joblib")
encoder = joblib.load("encoder_elm_heart.joblib")

st.title("💖 Prediksi Penyakit Jantung dengan ELM")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko terkena penyakit jantung.")

# Input data
umur = st.number_input("Umur", 1, 120, 40)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
tipe_nyeri_dada = st.number_input("Tipe Nyeri Dada (0–3)", 0, 3, 0)
tekanan_darah = st.number_input("Tekanan Darah (mmHg)", 50, 250, 120)
kolesterol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
gula_darah = st.selectbox("Gula Darah > 120 mg/dl (1=Ya,0=Tidak)", [0, 1])
hasil_ecg = st.number_input("Hasil ECG (0–2)", 0, 2, 0)
denyut_jantung = st.number_input("Denyut Jantung Maksimum", 60, 250, 150)
angina_olahraga = st.selectbox("Angina karena olahraga (1=Ya,0=Tidak)", [0, 1])
st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope_st = st.number_input("Slope ST (0–2)", 0, 2, 1)
pembuluh_besar = st.number_input("Jumlah pembuluh besar (0–4)", 0, 4, 0)
thalassemia = st.number_input("Thalassemia (0–3)", 0, 3, 1)

# Konversi input ke numpy array
data_input = np.array([[
    umur,
    1 if jenis_kelamin == "Laki-laki" else 0,
    tipe_nyeri_dada,
    tekanan_darah,
    kolesterol,
    gula_darah,
    hasil_ecg,
    denyut_jantung,
    angina_olahraga,
    st_depression,
    slope_st,
    pembuluh_besar,
    thalassemia
]])

# Prediksi
if st.button("Prediksi"):
    try:
        pred = model.predict(data_input)
        hasil_label = encoder.inverse_transform(pred.reshape(-1, 1))[0][0]
        if hasil_label == 1:
            st.error("⚠️ Pasien **berisiko tinggi** terkena penyakit jantung.")
        else:
            st.success("✅ Pasien **tidak berisiko** penyakit jantung.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
