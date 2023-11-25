import requests
import json

def infer_vm(text, lang ):
    url = "http://127.0.0.1:8000/inferV"

    payload = json.dumps({
    "text": text,
    "user_id": "voice_maker",
    "project_id": "try_vm_en_base",
    "language": lang,
    "gender": "male"
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

if __name__ == "__main__":
    infer_vm("সবাই বলছে ভারত ধীরে ধীরে চীনকে পেছনে ফেলে বিশ্বের উৎপাদন কেন্দ্র।", "bn-IN")