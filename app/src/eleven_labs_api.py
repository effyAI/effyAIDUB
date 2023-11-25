import requests
import json

class ElevenAPI():
    def __init__(self):
        self.url = "https://api.elevenlabs.io/"
        self.api_key = ""


    def list_voices(self):
        endpoint = self.url + "v1/voices"
        
        payload = {}
        headers = {
            'xi-api-key': self.api_key
        }
        
        response = requests.request("GET", endpoint, headers=headers, data=payload)
        return response.json()

    def add_voice(self, name, src_voice_list):
        endpoint = self.url + "v1/voices/add"

        payload = {'name': name}

        files = []
        for svl in range(len(src_voice_list)):
            fmt = ('files',('{}.wav'.format(svl),open('{}'.format(src_voice_list[svl]),'rb'),'audio/wav'))
            files.append(fmt)
        
        headers = {
            'xi-api-key': self.api_key
        }

        response = requests.request("POST", endpoint, headers=headers, data=payload, files=files)

        return response.json()

    def tts(self, voice_id, text, save_path, model = "eleven_multilingual_v2"):
        endpoint = self.url + "v1/text-to-speech/{}".format(voice_id) #/v1/text-to-speech/{voice_id}/stream

        payload = json.dumps({
            "text": text,
            "model_id": model,
            "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75}
        })

        headers = {
            'Content-Type': 'application/json',
            'xi-api-key': self.api_key
        }

        response = requests.request("POST", endpoint, headers=headers, data=payload)

        with open(save_path, "wb") as file:
            file.write(response.content)
    
    def get_voice_id(self, name):
        voice_list = self.list_voices()
        for vl in voice_list['voices']:
            if name == vl['name']:
                return vl['voice_id']
        return ''
    
    def delete_voice(self, voice_id):
        endpoint = self.url + "v1/voices/{}".format(voice_id)

        payload = {}
        headers = {
            'xi-api-key': self.api_key
        }

        response = requests.request("DELETE", endpoint, headers=headers, data=payload)
        print(response.status_code)

        return response