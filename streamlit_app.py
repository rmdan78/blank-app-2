import streamlit as st
import pandas as pd
import pickle

# Memuat model
model_path = 'model_stress_gptv2.pkl'
try:
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        expected_features = model_data['feature_names']
except FileNotFoundError:
    st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang benar.")
    st.stop()
except pickle.UnpicklingError:
    st.error(f"File model '{model_path}' rusak atau tidak dapat dibaca.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Modul yang diperlukan oleh model tidak ditemukan: {e}")
    st.stop()
except KeyError:
    st.error("Model tidak berisi informasi tentang nama fitur. Pastikan model dilatih dengan nama fitur yang disimpan.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

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
        'Sleep Quality': [sleep_quality],
        'Headaches Frequency': [headaches_frequency],
        'Academic Performance': [academic_performance],
        'Study Load': [study_load],
        'Extracurricular Activities': [extracurricular_activities]
    }
    df_baru = pd.DataFrame(data_baru)
    
    # Memastikan kolom sesuai dengan yang diharapkan oleh model
    if list(df_baru.columns) != expected_features:
        st.error(f"Kolom data tidak sesuai. Harap pastikan kolom data adalah: {expected_features}")
    else:
        try:
            # Membuat prediksi
            prediksi = model.predict(df_baru)
            # Menampilkan hasil prediksi
            st.success(f"Tingkat Stres Anda: {prediksi[0]}")
        except ValueError as e:
            st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan yang tidak terduga: {e}")
