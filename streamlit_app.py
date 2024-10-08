import streamlit as st
import pandas as pd
import pickle

# Memuat model
model_path = 'model_stress (2).pkl'
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
st.title("Prediksi Tingkat Stres Mahasiswa/Pelajar")

# Membuat form input
st.header("Masukkan Data")
# Input data

st.markdown("Silakan Nilai Kualitas Tidur Anda 😴", unsafe_allow_html=True)
sleep_quality = st.number_input('1 = Pulas, 2 = Cukup Pulas, 3 = Biasa Saja, 4 = Kurang Pulas, 5 = Tidak Pulas', min_value=1, max_value=5, step=1)
st.markdown("Berapa kali dalam seminggu Anda menderita sakit kepala 🤕?", unsafe_allow_html=True)
headaches_frequency = st.number_input(' 1 = Tidak pernah, 2 = 1 s/d 3, 3 = 4 s/d 6, 4 = 7 s/d 9, 5 = Lebih Dari 10', min_value=1, max_value=5, step=1)
st.markdown("Bagaimana Anda menilai kinerja akademik Anda 👩‍🎓?", unsafe_allow_html=True)
academic_performance = st.number_input('1 = Bagus, 2 = Cukup Bagus, 3 = Cukup, 4 = Kurang Bagus, 5 = Tidak Bagus', min_value=1, max_value=5, step=1)
st.markdown("Bagaimana Anda menilai beban studi Anda?", unsafe_allow_html=True)
study_load = st.number_input('1 = Ringan, 2 = Cukup Ringan, 3 = Biasa, 4 = Cukup Berat, 5 = Berat', min_value=1, max_value=5, step=1)
st.markdown("Berapa kali dalam seminggu Anda berlatih kegiatan ekstrakurikuler 🎾?", unsafe_allow_html=True)
extracurricular_activities = st.number_input('1 = Tidak pernah, 2 = 1 s/d 3,  2 = 4 s/d 6,  4 = 7 s/d 9, 5 = Lebih Dari 10', min_value=1, max_value=5, step=1)

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
            st.markdown("Prediksi tingkat stres 1 s/d 5", unsafe_allow_html=True)
            st.success(f"Tingkat Stres Anda: {prediksi[0]}")
        except ValueError as e:
            st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan yang tidak terduga: {e}")
