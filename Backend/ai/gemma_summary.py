from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

def generate_gemma_summary(results_list):
    if not results_list:
        return "Son 1 dakikada analiz yok."
    total = len(results_list)
    tumor_cases = [r for r in results_list if r['predicted_class'] != "no_tumor"]
    tumor_count = len(tumor_cases)

    # Tespit edilen tümör tiplerini (ör: glioma, meningioma, pituitary) çıkar
    tumor_types = [r['predicted_class'] for r in tumor_cases]
    if tumor_types:
        types_text = ", ".join(set(tumor_types))
    else:
        types_text = "Yok"

    prompt = (
        f"{total} MR görüntüsü incelendi. "
        f"{tumor_count} tanesinde tümör tespit edildi. "
        f"Bulunan tümör türleri: {types_text}. "
        "tümörlü örnek sayısını belirt, hangi tiplerin bulunduğunu kısa ve açık yaz, genel bir klinik yorum ekle. "
        "Yanıtın Türkçe ve en fazla 3-4 cümle olsun."
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it",  # <-- Burada mutlaka bu modeli kullan!
        messages=messages,
        max_tokens=80
    )
    return completion.choices[0].message.content.strip()
