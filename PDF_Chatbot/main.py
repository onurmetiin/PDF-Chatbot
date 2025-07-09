# -- A# Soruya en uygun yanıtı Google Gemini LLM ile oluşturmak ve ekranda göstermekA UYGULAMA DOSYASI (STREAMLİT ARAYÜZÜ) --

# Kulanıcıdan PDF dosyasını almak
# PDF'ten metin çıkarma ve parçalama işlemini başlatmak
# Vektör veritabanı oluşturma
# Kullanıcıdan bir soru almak
# Soruya en uygun yanıtı OpenAI LLM ile oluşturmak ve ekranda göstermek

import streamlit as st  # Streamlit modülü (Web uygulaması için arayüz kütüphanesi)
from pdf_utils import extract_text_from_pdf, split_text_into_chunks  # PDF okuma ve metin bölme fonksiyonları
from rag_pipeline import create_vector_db, ask_question  # RAG (Retrieval-Augmented Generation) işlemleri (Embedding oluşturma ve GPT üzerinden yanıt alma)

# Streamlit uygulaması için sayfa yapılandırması
st.set_page_config(page_title="📄 PDF ChatBot", layout="wide")
st.title("📄 Chat with your PDF document!")  # Uygulama başlığı
st.caption("PDF Document-based artificial intelligence assistant powered by Google Gemini") # Uygulama açıklaması


# Kullanıcıdan PDF dosyası yüklemesi istenir
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")  # Kullanıcıdan PDF dosyası yüklemesi istenir
if uploaded_file is not None:
    st.success("Your PDF file uploaded!")  # Dosya yüklendiğinde başarı mesajı gösterilir
    
    
    # PDF dosyası yüklendiğinde metin çıkarma ve vektör veritabanı oluşturma işlemleri başlatılır
    with st.spinner("PDF işleniyor..."): #Kullanıcıya yükleniyor animasyonu gösterir
    
        text = extract_text_from_pdf(uploaded_file)  # PDF dosyasından metin çıkarılır
        chunks = split_text_into_chunks(text) # Metin, belirli boyutlarda parçalara bölünür
        vector_db = create_vector_db(chunks)  # Metin parçalarından vektör veritabanı oluşturulur
    
    
    # Kullanıcıdan bir soru girmesi istenir
    question = st.text_input("📥 What do you want to learn about in this file?")
    if question:
        with st.spinner("Document processing..."):
            answer = ask_question(vector_db, question)
            st.markdown(f"Answer: \n\n {answer}")  # Google Gemini LLM tarafından oluşturulan yanıt ekranda gösterilir        
        
        
# Uygulama çalıştırmak için terminalde şu komutu kullanın:
# streamlit run main.py