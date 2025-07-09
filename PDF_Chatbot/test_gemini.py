#!/usr/bin/env python3
# Test script to verify Google Gemini integration

import os
from dotenv import load_dotenv
from rag_pipeline import create_vector_db, ask_question

# Load environment variables
load_dotenv()

def test_gemini_integration():
    """Test Google Gemini integration with a simple example"""
    
    # Test text chunks
    test_chunks = [
        "Bu bir test belgesidir. PDF ChatBot projesi Google Gemini ile çalışmaktadır.",
        "Google Gemini, gelişmiş bir yapay zeka modelidir ve doğal dil işleme yetenekleri vardır.",
        "Bu proje PDF dosyalarından metin çıkararak kullanıcı sorularını yanıtlar."
    ]
    
    print("🔄 Test başlatılıyor...")
    print("📄 Test vektör veritabanı oluşturuluyor...")
    
    try:
        # Create vector database
        vector_db = create_vector_db(test_chunks)
        print("✅ Vektör veritabanı başarıyla oluşturuldu!")
        
        # Test question
        test_question = "Bu proje hangi AI modeli kullanıyor?"
        print(f"❓ Test sorusu: {test_question}")
        
        # Get answer
        answer = ask_question(vector_db, test_question)
        print(f"🤖 Gemini'nin yanıtı: {answer}")
        
        print("\n✅ Test başarıyla tamamlandı! Google Gemini entegrasyonu çalışıyor.")
        
    except Exception as e:
        print(f"❌ Test başarısız: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_gemini_integration()
    if success:
        print("\n🎉 Tüm testler başarılı! Uygulamanız Google Gemini ile çalışmaya hazır.")
    else:
        print("\n❌ Test başarısız. Lütfen konfigürasyonu kontrol edin.")
