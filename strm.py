import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Judul aplikasi
st.title("Diabetes Prediction using SVM with User-uploaded Dataset")

# Upload file CSV dataset
st.write("### Upload your diabetes dataset (CSV format)")

# Upload file CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Menampilkan dataset yang diupload
    st.write("Dataset Preview:")
    st.write(df.head())

    # Memeriksa kolom yang ada di dataset
    st.write("Columns in Dataset:")
    st.write(df.columns)

    # Memastikan bahwa ada kolom "Hasil" untuk target (output)
    if 'Hasil' in df.columns:
        # Memisahkan fitur dan target
        X = df.drop('Hasil', axis=1)  # Fitur
        y = df['Hasil']  # Target (apakah diabetes atau tidak)

        # Standarisasi fitur
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Membagi data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Membuat model SVM
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)

        # Prediksi dan akurasi
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan akurasi model
        st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

        # Form input untuk pengguna untuk prediksi baru
        st.write("### Input Features for Diabetes Prediction")

        input_data = []
        for column in X.columns:
            value = st.number_input(f"{column}", min_value=0, value=0)
            input_data.append(value)

        # Prediksi jika tombol ditekan
        if st.button("Predict"):
            # Mengubah input menjadi array dan standarisasi
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Prediksi hasil
            prediction = svm_model.predict(input_scaled)

            # Menampilkan hasil prediksi
            if prediction[0] == 1:
                st.write("### Predicted Result: Diabetes Risk (Yes)")
            else:
                st.write("### Predicted Result: No Diabetes Risk")
    else:
        st.write("### Error: Kolom 'Hasil' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom target 'Hasil'.")
