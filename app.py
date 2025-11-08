import streamlit as st
import joblib
import numpy as np

# === Load Model & Encoder ===
model = joblib.load("model_elm_heart.joblib")
encoder = joblib.load("encoder_elm_heart.joblib")

# === Judul Aplikasi ===
st.title("💖 Prediksi Penyakit Jantung dengan ELM")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko terkena penyakit jantung.")

# === Input Data Pengguna ===
umur = st.number_input("Umur", min_value=1, max_value=120, value=45)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
if jenis_kelamin == "Laki-laki":
    jenis_kelamin = 1
else:
    jenis_kelamin = 0

tipe_nyeri = st.number_input("Tipe Nyeri Dada (0–3)", min_value=0, max_value=3, value=1)
tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=50, max_value=250, value=120)
kolesterol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
gula_darah = st.selectbox("Gula Darah > 120 mg/dl (1=Ya,0=Tidak)", [0, 1])
hasil_ecg = st.number_input("Hasil ECG (0–2)", min_value=0, max_value=2, value=1)
denyut_jantung = st.number_input("Denyut Jantung Maksimum", min_value=50, max_value=250, value=150)
angina_olahraga = st.selectbox("Angina karena olahraga (1=Ya,0=Tidak)", [0, 1])
st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope_st = st.number_input("Slope ST (0–2)", min_value=0, max_value=2, value=1)
jumlah_pembuluh = st.number_input("Jumlah pembuluh besar (0–4)", min_value=0, max_value=4, value=0)
thalassemia = st.number_input("Thalassemia (0–3)", min_value=0, max_value=3, value=1)

# === Tombol Prediksi ===
if st.button("🩺 Prediksi Sekarang"):
    try:
        # Susun input jadi array
        data_input = np.array([[umur, jenis_kelamin, tipe_nyeri, tekanan_darah, kolesterol,
                                gula_darah, hasil_ecg, denyut_jantung, angina_olahraga,
                                st_depression, slope_st, jumlah_pembuluh, thalassemia]])

        # Prediksi menggunakan model
        pred = model.predict(data_input)

        # Jika encoder output bentuknya 1 kolom, lakukan inverse transform
        try:
            hasil = encoder.inverse_transform(pred.reshape(-1, 1))[0][0]
        except Exception:
            hasil = pred[0]

        # Tampilkan hasil
        if hasil == 1 or hasil == "1":
            st.error("⚠️ Pasien **berisiko tinggi** terkena penyakit jantung.")
        else:
            st.success("✅ Pasien **tidak berisiko** penyakit jantung.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
