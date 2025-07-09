# -- PDF OKUMA, METİN ÇIKARMA VE BÖLME FONKSİYONLARI -- 

import fitz  # PyMuPDF modülü
from langchain.text_splitter import CharacterTextSplitter # Metin bölme modülü, belirli karakter sayısına göre parçalar

def extract_text_from_pdf(file):
    """
    PDF dosyasından metin çıkarır.
    
    Args:
        file: PDF dosyasının yolu veya dosya nesnesi.
    
    Returns:
        str: PDF dosyasından çıkarılan metin.
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")  # PDF dosyasını aç
    text = ""
    
    for page in doc:  # Her sayfayı döngüye al
        text += page.get_text()  # Sayfadaki metni al ve birleştir
        
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """
    Metni belirli boyutlarda parçalara böler.
    
    Args:
        text (str): Parçalanacak metin.
        chunk_size (int): Her parçanın maksimum karakter sayısı.
        chunk_overlap (int): Parçalar arasındaki karakter örtüşmesi. Bir parça diğerine kaç karakter ortaklıkla bağlansın. Amaç: bağlam kaybını önlemek.
    
    Returns:
        list: Parçalanmış metin listesi.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap )  # Metin bölücü oluştur
    
    return text_splitter.split_text(text)  # Metni parçala ve liste olarak döndür