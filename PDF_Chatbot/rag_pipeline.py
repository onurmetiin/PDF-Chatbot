# -- EMBEDDING(VEKTÖR) OLUSTURMA, VEKTÖR DB KURMA, KULLANICI SORGU İŞLEMLERİ --

# Metin parçalarından embedding'ler oluşturur, (vektör).
# Bu vektörleri FAISS veritabanında saklar.
# Kullanıcı sorgularını işler ve en yakın vektörleri bulur.
# Google Gemini LLM ile en uygun yanıtı oluşturur.

from dotenv import load_dotenv
import os
load_dotenv()  # .env dosyasındaki API anahtarlarını otomatik yükler

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # Google Gemini modülleri
from langchain_community.vectorstores import FAISS  # FAISS vektör veritabanı

def create_vector_db(chunks):
    """
    Metin parçalarından embedding (vektör veritabanı) oluşturur.
    
    Args:
        chunks (list[str]): split_text_into_chunks() fonksiyonu ile elde edilen metin parçaları
    
    Returns:
        FAISS: Oluşturulan FAISS vektör veritabanı. LangChain'in FAISS nesnesi (vektör arama için)
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )  # Google embedding nesnesi oluştur
    vector_db = FAISS.from_texts(chunks, embeddings)  # Parçalardan vektör veritabanı oluştur
    return vector_db


def ask_question(vector_db, question, model_name="gemini-1.5-flash"):
    """
    Kullanıcı sorgusunu işler ve en yakın vektörleri bulur.
    
    Args:
        vector_db (FAISS): Vektör veritabanı.
        question (str): Kullanıcının sorusu.
        model_name (str): Google Gemini model adı (varsayılan: "gemini-1.5-flash").
    
    Returns:
        str: Google Gemini LLM tarafından oluşturulan yanıt.
    """
    # 1. Benzer metinleri bul (retrieval)
    docs = vector_db.similarity_search(question)

    # 2. Gemini modelini başlat
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    # 3. Bağlam oluştur
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 4. Prompt oluştur ve yanıt al
    prompt = f"""
    Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtlayın. 
    Eğer bağlam bilgilerinde yeterli bilgi yoksa, "Bu konuda yeterli bilgi bulunamadı" şeklinde yanıt verin.
    
    Bağlam:
    {context}
    
    Soru: {question}
    
    Yanıt:
    """
    
    # 5. Yanıt üret
    response = llm.invoke(prompt)
    
    return response.content
