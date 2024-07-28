import streamlit as st
import pandas as pd
import pickle

# Muat model

# Memastikan semua modul yang diperlukan terinstal

model = pickle.load(open('model_stress (1).pkl', 'rb'))
# Judul aplikasi
st.title("Form Prediksi Tingkat Stres")

# Membuat form input
st.header("Masukkan Data")

# Input data
sleep_quality = st.number_input('Kindly Rate your Sleep Quality 😴', min_value=1, max_value=5, step=1)
headaches_frequency = st.number_input('How many times a week do you suffer headaches 🤕?', min_value=0, max_value=7, step=1)
academic_performance = st.number_input('How would you rate your academic performance 👩‍🎓?', min_value=1, max_value=5, step=1)
study_load = st.number_input('How would you rate your study load?', min_value=1, max_value=5, step=1)
extracurricular_activities = st.number_input('How many times a week you practice extracurricular activities 🎾?', min_value=0, max_value=7, step=1)

# Tombol untuk membuat prediksi
if st.button('Prediksi Tingkat Stres'):
    # Data baru
    data_baru = {
        'Kindly Rate your Sleep Quality 😴': [sleep_quality],
        'How many times a week do you suffer headaches 🤕?': [headaches_frequency],
        'How would you rate your academic performance 👩‍🎓?': [academic_performance],
        'How would you rate your study load?': [study_load],
        'How many times a week you practice extracurricular activities 🎾?': [extracurricular_activities]
    }
    df_baru = pd.DataFrame(data_baru)
    st.dataframe(df_baru)
    # Membuat prediksi
    prediksi = model.predict(df_baru)
        # Menampilkan hasil prediksi
    st.success(f"Tingkat Stres Anda: {prediksi[0]}")
    

# Menampilkan DataFrame
if 'dataframe' in st.session_state:
    st.header("DataFrame Sebelumnya")
    st.dataframe(st.session_state.dataframe)
else:
    st.info("Belum ada data yang dimasukkan.")
