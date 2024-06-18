import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import plotly.graph_objs as go
import os
import calendar

from wordcloud import WordCloud
from collections import Counter
from streamlit_option_menu import option_menu
from pymongo import MongoClient
from dotenv import load_dotenv

# Settings
st.set_page_config(page_title="Analisis Sentimen")

st.markdown("""
    <style>
        div.block-container {padding-top:2rem;}
        div.block-container {padding-bottom:2rem;}
    </style>
""", unsafe_allow_html=True)

# MongoDB Connection
load_dotenv(".env")
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
db = client[os.getenv("MONGO_DATABASE_NAME")]
collection = db[os.getenv("MONGO_COLLECTION_NAME")]

# Load Model
def load_model():
    return pickle.load(open('sentiment_analysis_model.pkl', 'rb'))

# Add Data
def add_data(username, access_control, name, password):
    # Check if the username already exists
    if collection.find_one({"username": username}):
        return {"status": "fail", "message": "Username already exists"}

    # If username doesn't exist, proceed to insert the new data
    data = {
        "username": username,
        "access_control": access_control,
        "name": name,
        "password": password
    }

    result = collection.insert_one(data)
    return {"status": "success", "inserted_id": str(result.inserted_id)}

# Delete Data
def delete_data(username):
    return collection.delete_one({"username": username})

# Update User
def update_user(username, access_control, name, password):
    collection.update_one(
        {"username": username},
        {"$set": {
            "access_control": access_control,
            "name": name,
            "password": password
        }}
    )

# Login
def validate_login(username, password):
    user = collection.find_one({"username": username, "password": password})
    return user

# Cache Data
@st.cache_data
def load_users_data():
    users_data = list(collection.find({}, {"_id": 0, "username": 1, "access_control": 1, "name": 1}))
    df_users = pd.DataFrame(users_data).astype(str)
    return df_users

# Clear Cache
def clear_cache():
    load_users_data.clear()

# Halaman Login
def login_page():
    st.markdown("<h1 style='text-align: center;'>Selamat Datang!</h1>", unsafe_allow_html=True)
    st.text("")

    col1, col2, col3 = st.columns([0.5, 1.5, 0.5])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username Anda")
            password = st.text_input("Password", placeholder="Masukkan password Anda", type="password")
            login_button = st.form_submit_button("Login", type="primary")

        if login_button:
            user = validate_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Username atau password salah. Silakan coba lagi.")

# Halaman About
def about_page():
    st.title('Analisis Sentimen Komentar Youtube Honda Menggunakan Metode Naive Bayes')
    st.write("Halaman ini bertujuan untuk menyajikan hasil analisis sentimen komentar pada kanal YouTube **'Welove Honda Indonesia'** pada video yang berjudul **'Klarifikasi Kemunculan Warna Kuning Pada Rangka Honda'**. Data diambil dari komentar netizen, dengan total data setelah dilakukan preprocessing mencapai **2240 data**. Kami melakukan pelabelan otomatis menggunakan metode **Lexicon Based** yang membagi data menjadi dua kelas, yaitu **Positif dan Negatif**.")

    # Membaca file Excel
    data = pd.read_excel('hasil_labeling_lexicon_manual.xlsx')

    st.write("**Data Labeling**:")
    st.dataframe(data, use_container_width=True)

    # Mendapatkan jumlah Label Positif, Netral, dan Negatif
    label_counts_lexicon = data['sentiment_label_lexicon'].value_counts()
    label_counts_manual = data['sentiment_label_manual'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        # Visualisasi Grafik Pie untuk Lexicon Sentiment
        st.subheader('Grafik Pie untuk Sentimen Label (Lexicon)')
        fig_pie_lexicon = go.Figure(
            data=[go.Pie(labels=label_counts_lexicon.index, values=label_counts_lexicon.values)])
        st.plotly_chart(fig_pie_lexicon, use_container_width=True)
    with col2:
        # Visualisasi Grafik Pie untuk Manual Sentiment
        st.subheader('Grafik Pie untuk Sentimen Label (Manual)')
        fig_pie_manual = go.Figure(data=[go.Pie(labels=label_counts_manual.index, values=label_counts_manual.values)])
        st.plotly_chart(fig_pie_manual, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Visualisasi Grafik Batang untuk Lexicon Sentiment
        st.subheader('Grafik Batang untuk Sentimen Label (Lexicon)')
        fig_bar_lexicon = go.Figure(data=[go.Bar(x=label_counts_lexicon.index, y=label_counts_lexicon.values)])
        fig_bar_lexicon.update_xaxes(title='Sentiment Label (Lexicon)')
        fig_bar_lexicon.update_yaxes(title='Count')
        st.plotly_chart(fig_bar_lexicon, use_container_width=True)
    with col2:
        # Visualisasi Grafik Batang untuk Manual Sentiment
        st.subheader('Grafik Batang untuk Sentimen Label (Manual)')
        fig_bar_manual = go.Figure(data=[go.Bar(x=label_counts_manual.index, y=label_counts_manual.values)])
        fig_bar_manual.update_xaxes(title='Sentiment Label (Manual)')
        fig_bar_manual.update_yaxes(title='Count')
        st.plotly_chart(fig_bar_manual, use_container_width=True)

    # Filter teks berdasarkan sentimen
    # Step 1: Convert the 'text' column to strings
    data['text'] = data['text'].astype(str)
    st.write("**Create Word Cloud**:")
    st.dataframe(data, use_container_width=True)
    # Adding a selectbox for choosing the sentiment label type
    label = st.selectbox('Pilih jenis label sentimen:', ['sentiment_label_lexicon', 'sentiment_label_manual'])
    # Assuming you want to use the lexicon sentiment labels for the word clouds
    positive_text = ' '.join(data[data[label] == 'Positif']['text'])
    negative_text = ' '.join(data[data[label] == 'Negatif']['text'])
    all_text = ' '.join(data['text'])

    # Visualisasi Wordcloud untuk sentimen positif
    st.subheader('Wordcloud untuk Sentimen Positif')
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    st.image(wordcloud_positive.to_array(), caption='Wordcloud Sentimen Positif', use_column_width=True)

    # Visualisasi Wordcloud untuk sentimen negatif
    st.subheader('Wordcloud untuk Sentimen Negatif')
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    st.image(wordcloud_negative.to_array(), caption='Wordcloud Sentimen Negatif', use_column_width=True)

    # Visualisasi Wordcloud untuk semua kata
    st.subheader('Wordcloud untuk Semua Kata')
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), caption='Wordcloud Semua Kata', use_column_width=True)

    # Function to create bar charts for word frequencies
    def create_word_freq_bar_chart(text, title, color):
        words = text.split()
        word_freq = Counter(words)
        top_words = word_freq.most_common(10)
        words, freq = zip(*top_words)
        fig = go.Figure([go.Bar(x=words, y=freq, marker=dict(color=color))])
        fig.update_layout(title=title, xaxis_title='Kata', yaxis_title='Frekuensi')
        return fig

    # Sentimen positif
    fig_positive = create_word_freq_bar_chart(positive_text, '10 Kata yang Paling Sering Muncul (Sentimen Positif)',
                                              'green')
    # Sentimen negatif
    fig_negative = create_word_freq_bar_chart(negative_text, '10 Kata yang Paling Sering Muncul (Sentimen Negatif)',
                                              'red')
    # Semua kata
    fig_all = create_word_freq_bar_chart(all_text, '10 Kata yang Paling Sering Muncul (Semua)', 'blue')

    # Menampilkan grafik menggunakan Plotly
    st.plotly_chart(fig_positive, use_container_width=True)
    st.plotly_chart(fig_negative, use_container_width=True)
    st.plotly_chart(fig_all, use_container_width=True)

    # Menambahkan kalimat tambahan
    st.markdown("""
        Setelah dilakukan pelabelan dengan lexicon based dan secara manual, selanjutnya kami melakukan feature extractor dengan **TF-IDF**. Selanjutnya kami melakukan klasifikasi dengan **Complement Naive Bayes**. 
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            Berikut adalah hasil evaluasi label Lexicon:
    
            - **Akurasi Complement Naive Bayes:** 79.41%
            - **Presisi Complement Naive Bayes:** 79.29%
            - **Recall Complement Naive Bayes:** 79.41%
            - **F1-score Complement Naive Bayes:** 79.08%
        """)
    with col2:
        st.markdown("""
            Berikut adalah hasil evaluasi label manual:
    
            - **Akurasi Complement Naive Bayes:** 75.79%
            - **Presisi Complement Naive Bayes:** 73.97%
            - **Recall Complement Naive Bayes:** 75.79%
            - **F1-score Complement Naive Bayes:** 71.93%
        """)

# Halaman Predict Text
def predict_text_page(model):
    st.subheader("Predict Sentiment from Text")
    tweet = st.text_input('Enter your tweet')
    submit = st.button('Predict')

    if submit:
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
        st.write('Predicted Sentiment:', prediction[0])

# Halaman Predict DataFrame
def predict_dataframe_page(model):
    st.subheader("Predict Sentiment from DataFrame")
    uploaded_file = st.file_uploader("Upload your DataFrame", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' else pd.read_csv(uploaded_file)
        st.write('Uploaded DataFrame:')
        st.write(df)
        # Create a list of column headers from the DataFrame
        column_df = df.columns.tolist()
        # Add a select box for choosing the 'access_control' column
        selectbox = st.selectbox("Select Features", column_df, placeholder="Pilih kolom yang akan di prediksi", index=None)
        submit_df = st.button(f'Predict Column {selectbox} Sentiments from DataFrame')

        if submit_df:
            start = time.time()
            # Handle NaN values in the selected column
            selected_column = df[selectbox].dropna()  # Remove NaN values
            predictions = model.predict(selected_column)
            end = time.time()
            st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
            st.write('Predicted Sentiments:')

            predictions_df = pd.DataFrame(predictions, index=selected_column.index, columns=['Predicted Sentiments'])
            # Concatenate the original column with the predictions
            result_df = pd.concat([selected_column, predictions_df], axis=1)
            st.dataframe(result_df)

            # Menambahkan grafik pie
            sentiment_counts = pd.Series(predictions).value_counts()
            st.subheader('Sentiment Distribution')

            # Membuat data untuk grafik pie
            labels = sentiment_counts.index
            values = sentiment_counts.values

            # Menentukan palet warna
            palette_colors = ['#be185d', '#500724']

            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values,  marker=dict(colors=palette_colors))])
            st.plotly_chart(fig_pie, use_container_width=True)

            # Menambahkan grafik bar
            st.subheader('Sentiment Counts')

            fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=palette_colors)])
            fig_bar.update_layout(xaxis_title='Sentiment', yaxis_title='Count')
            st.plotly_chart(fig_bar, use_container_width=True)

# Halaman Access Management Admin
def access_management_page_admin():
    with st.sidebar:
        access_management_page_option = st.selectbox("Menu Access Management", ["Add User", "Delete User", "Edit User"])

    users_data = load_users_data()

    with st.container(border=True):
        st.subheader("Users Data")
        users_df = st.dataframe(users_data, use_container_width=True)

    st.divider()

    if access_management_page_option == "Add User":
        with st.form("add_user_form", clear_on_submit=True):
            st.subheader("Form Add User")

            add_username = st.text_input("Username", placeholder="Masukkan username")
            add_access_control = st.selectbox("Access Control", ["Admin", "User"], placeholder="Pilih access control", index=None)
            add_name = st.text_input("Nama", placeholder="Masukkan nama")
            add_password = st.text_input("Password", placeholder="Masukkan password", type="password")
            add_user_button = st.form_submit_button("Add User", type="primary")

            if add_user_button:
                if add_username and add_access_control and add_name and add_password:
                    result = add_data(add_username, add_access_control, add_name, add_password)
                    if result['status'] == 'success':
                        message_success = st.success(f"User {add_name} telah berhasil ditambahkan")
                        time.sleep(3)
                        message_success.empty()

                        clear_cache()
                        users_data = load_users_data()
                        users_df.data = users_data
                        st.experimental_rerun()
                    else:
                        message_error = st.error(result['message'])
                        time.sleep(3)
                        message_error.empty()
                else:
                    message_error = st.error("Harap isi semua field")
                    time.sleep(3)
                    message_error.empty()

    elif access_management_page_option == "Delete User":
        with st.form("delete_user_form", clear_on_submit=True):
            st.subheader("Form Delete User")

            delete_username = st.selectbox("Username", np.sort(pd.DataFrame(users_data)["username"].unique()), placeholder="Pilih username", index=None)
            delete_user_button = st.form_submit_button("Delete User", type="primary")

            if delete_user_button:
                if delete_username:
                    delete_data(delete_username)
                    message_success = st.success(f"User {delete_username} telah berhasil dihapus")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap pilih username")
                    time.sleep(3)
                    message_error.empty()

    else:
        with st.form("edit_user_form", clear_on_submit=True):
            st.subheader("Form Edit User")

            edit_username = st.selectbox("Username", np.sort(pd.DataFrame(users_data)["username"].unique()), placeholder="Pilih username", index=None)
            edit_access_control = st.selectbox("Access Control", ["Admin", "User"], placeholder="Pilih access control", index=None)
            edit_name = st.text_input("Nama", placeholder="Masukkan nama")
            edit_password = st.text_input("Password", placeholder="Masukkan password", type="password")
            edit_user_button = st.form_submit_button("Edit User", type="primary")

            if edit_user_button:
                if edit_username and edit_access_control and edit_name and edit_password:
                    update_user(edit_username, edit_access_control, edit_name, edit_password)
                    message_success = st.success(f"User {edit_username} telah berhasil diperbarui")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap isi semua field")
                    time.sleep(3)
                    message_error.empty()

# Halaman Access Management User
def access_management_page_user():
    current_user = st.session_state.get("user", {})
    edit_username = current_user.get("username", "")
    edit_access_control = current_user.get("access_control", "User")

    with st.form("edit_user_form_user", clear_on_submit=True):
        st.subheader("Form Edit User")

        edit_name = st.text_input("Username", placeholder="Username", value=edit_username)
        edit_password = st.text_input("Password", placeholder="Masukkan password", type="password")
        edit_user_button = st.form_submit_button("Edit User", type="primary")

        if edit_user_button:
            if edit_name and edit_password:
                update_user(edit_username, edit_access_control, edit_name, edit_password)
                message_success = st.success(f"User {edit_username} telah berhasil diperbarui")
                time.sleep(3)
                message_success.empty()

                clear_cache()
                st.session_state.user["name"] = edit_name
                st.rerun()
            else:
                message_error = st.error("Harap isi semua field")
                time.sleep(3)
                message_error.empty()

# Halaman Report
def report_page():
    st.subheader("Report Dataset Komentar Youtube Honda")

    report_data = pd.read_excel("Klarifikasi_Kemunculan_Warna_Kuning_Pada_Rangka_Honda.xlsx")
    report_data["pubdate"] = pd.to_datetime(report_data["pubdate"])

    col1, col2 = st.columns(2)
    with col1:
        year_filter = st.selectbox(
            "Tahun",
            np.sort(report_data["pubdate"].dt.year.unique()),
            placeholder="Pilih Tahun",
            index=None
        )

    with col2:
        if year_filter:
            year_filtered_report_data = report_data[
                report_data["pubdate"].dt.year == year_filter
            ].reset_index(drop=True)

            unique_months = np.sort(year_filtered_report_data["pubdate"].dt.month.unique())
            month_names = [calendar.month_name[month] for month in unique_months]

            month_filter = st.selectbox(
                "Bulan",
                month_names,
                placeholder="Pilih Bulan",
                index=None
            )

            if month_filter:
                month_number = {name: num for num, name in enumerate(calendar.month_name) if num in unique_months}[month_filter]

                year_month_filtered_report_data = year_filtered_report_data[
                    year_filtered_report_data["pubdate"].dt.month == month_number
                ].reset_index(drop=True)
        else:
            month_filter = st.selectbox(
                "Bulan",
                ["Pilih tahun terlebih dahulu"],
                placeholder="Pilih Bulan",
                index=None
            )

    if year_filter and month_filter:
        st.dataframe(year_month_filtered_report_data, use_container_width=True)
    elif year_filter:
        st.dataframe(year_filtered_report_data, use_container_width=True)
    else:
        st.dataframe(report_data, use_container_width=True)

# Session State untuk Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Tampilkan Halaman Login Jika Belum Login
if not st.session_state.logged_in:
    login_page()
else:
    # Tampilkan Halaman Isi
    with st.sidebar:
        user = st.session_state.get("user", {})
        user_name = user.get("name", "User")
        user_access = user.get("access_control", "user")
        st.markdown(f"Selamat datang, {user_access} {user_name}")

        menu_title = "Menu Admin" if user_access.lower() == "admin" else "Menu User"
        menu_options = ["About", "Predict Text", "Predict DataFrame", "Access Management"]

        if user_access.lower() == "admin":
            menu_options.append("Report")

        option = option_menu(
            menu_title,
            menu_options,
            menu_icon="cast",
            default_index=0
        )

        logout_button = st.button("Logout", type="primary")

    # Logout
    if logout_button:
        st.session_state.logged_in = False
        st.rerun()

    # Menentukan Halaman yang Sesuai
    if option == "About":
        about_page()
    elif option == "Predict Text":
        model = load_model()
        predict_text_page(model)
    elif option == "Predict DataFrame":
        model = load_model()
        predict_dataframe_page(model)
    elif option == "Access Management":
        if user_access.lower() == "admin":
            access_management_page_admin()
        else:
            access_management_page_user()
    else:
        report_page()
