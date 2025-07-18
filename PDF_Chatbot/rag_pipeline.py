# -- EMBEDDING(VEKTOR) OLUSTURMA, VEKTÖR DB KURMA, KULLANICI SORGU İŞLEMLERİ --

# Metin parçalarından embedding'ler oluşturur, (vektör).
# Bu vektörleri FAISS veritabanında saklar.
# Kullanıcı sorgularını işler ve en yakın vektörleri bulur.
# OpenAI LLM ile en uygun yanıtı oluşturur.

from dotenv import load_dotenv
load_dotenv()  # .env dosyasındaki API anahtarlarını otomatik yükler

from langchain_community.embeddings import OpenAIEmbeddings  # OpenAI embedding modülü
from langchain_community.vectorstores import FAISS  # FAISS vektör veritabanı
from langchain_community.chat_models import ChatOpenAI  # OpenAI sohbet modeli
from langchain.chains.question_answering import load_qa_chain  # Soru cevaplama zinciri

def create_vector_db(chunks):
    """
    Metin parçalarından embedding (vektör veritabanı) oluşturur.
    
    Args:
        chunks (list[str]): split_text_into_chunks() fonksiyonu ile elde edilen metin parçaları
    
    Returns:
        FAISS: Oluşturulan FAISS vektör veritabanı. LangChain'in FAISS nesnesi (vektör arama için)
    """
    embeddings = OpenAIEmbeddings()  # OpenAI embedding nesnesi oluştur
    vector_db = FAISS.from_texts(chunks, embeddings)  # Parçalardan vektör veritabanı oluştur
    return vector_db


def ask_question(vector_db, question, model_name="gpt-4", simularity_threshold=0.7):
    """
    Kullanıcı sorgusunu işler ve en yakın vektörleri bulur.
    
    Args:
        vector_db (FAISS): Vektör veritabanı.
        question (str): Kullanıcının sorusu.
        model_name (str): OpenAI model adı (varsayılan: "gpt-4").
    
    Returns:
        str: OpenAI LLM tarafından oluşturulan yanıt.
    """
    # 1. Benzer metinleri bul (retrieval) OLD
    #docs = vector_db.similarity_search(question) OLD

    # 1. Bağlamla score'u yüksek benzer metinleri bul (retrieval)
    results = vector_db.similarity_search_with_score(question, k=3) # k=3, en yakın 3 metin parçasını getirir.
    
    # Eğer benzerlik skoru belirtilen eşik değerinden düşükse, uygun bir yanıt bulunamadıysa
    if not results or all(score < simularity_threshold for _, score in results):
        return "Üzgünüm, sorunuza uygun bir yanıt bulamadım. Lütfen başka bir soru sorun."
    
    # simularity_threshold=0.7 score'una uyan verileri al
    revelant_docs = [doc for doc, score in results if score >= simularity_threshold]
    
    # 2. GPT modelini başlat
    llm = ChatOpenAI(model_name=model_name, tempreature = 0)

    # 3. Soru-cevap zinciri oluştur
    chain = load_qa_chain(llm, chain_type="stuff") 
    #chain_type="stuff" tüm parçaları birleştirip GPT'ye “tek parça” olarak verir. 
    # Daha gelişmiş alternatifleri: 
    # map_reduce: Uzun belgeler için önerilir. 
    # refine: İlk cevabı üretip diğer parçalarla geliştirir.

    # 4. En ilgili metin parçalarıyla yanıt üret
    return chain.run(
        input_documents=revelant_docs, 
        question=(question + "\n\nNot: Eğer yukarıdaki belgede bu soruya dair bilgi yoksa, lütfen uydurma bir yanıt vermeyin. Bunun yerine 'bu bilgi belgede bulunmuyor' gibi dürüst bir cevap verin.")
        )
