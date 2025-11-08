import numpy as np
import joblib
import streamlit as st

# === Tambahkan class ELMClassifier di sini ===
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
# === End class ===

# Sekarang baru load model
elm = joblib.load("model_elm_heart.joblib")
encoder = joblib.load("encoder_elm_heart.joblib")


st.title("💖 Prediksi Penyakit Jantung dengan ELM")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko terkena penyakit jantung.")

# === Input form ===
age = st.number_input("Umur", 1, 120, 50)
sex = st.selectbox("Jenis Kelamin", ("Perempuan", "Laki-laki"))
cp = st.number_input("Tipe Nyeri Dada (0–3)", 0, 3, 1)
trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 200, 120)
chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
fbs = st.number_input("Gula Darah > 120 mg/dl (1=Ya,0=Tidak)", 0, 1, 0)
restecg = st.number_input("Hasil ECG (0–2)", 0, 2, 1)
thalach = st.number_input("Denyut Jantung Maksimum", 60, 220, 150)
exang = st.number_input("Angina karena olahraga (1=Ya,0=Tidak)", 0, 1, 0)
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.number_input("Slope ST (0–2)", 0, 2, 1)
ca = st.number_input("Jumlah pembuluh besar (0–4)", 0, 4, 0)
thal = st.number_input("Thalassemia (0–3)", 0, 3, 1)

# Ubah gender ke numerik
sex_val = 1 if sex == "Laki-laki" else 0

# Susun input
X_input = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                     thalach, exang, oldpeak, slope, ca, thal]])

# === Prediksi ===
if st.button("Prediksi"):
    pred = elm.predict(X_input)
    hasil = encoder.inverse_transform(pred.reshape(-1, 1))[0][0]

    if hasil == 1:
        st.error("⚠️ Pasien **berisiko terkena penyakit jantung.**")
    else:
        st.success("✅ Pasien **tidak berisiko** terkena penyakit jantung.")
