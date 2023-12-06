import subprocess
import os
from tqdm import tqdm
# from so_vits_svc_fork.inference.main import 

import requests
import zipfile
import tempfile
import os
import json
import time
from pydub import AudioSegment

class AudioAPI:
    def __init__(self):
        self.auth_token = "Bearer b481e2c0-6eae-11ee-82e6-7bf8df377ea1"
        self.content_type = "application/json"
        self.voice_endpoint = "https://developer.voicemaker.in/voice/api"

    def get_audio_json(self, text, lang_id, voice_id):
        payload = json.dumps({
            "Engine": "neural",
            "VoiceId": voice_id,
            "LanguageCode": lang_id,
            "Text": text,
            "OutputFormat": "wav",
            "SampleRate": "48000",
            "Effect": "default",
            "MasterSpeed": "0",
            "MasterVolume": "0",
            "MasterPitch": "0"
        })

        print(payload)

        headers = {
            'Content-Type': self.content_type,
            'Authorization': self.auth_token,
            'Cookie': 'connect.sid=s%3A8NnmiVFc7nKY1_6kt7u6QjU7koVv8Bdr.foIcT6se%2BFulZHUF3F9zkLpEz3qmAJy2sxkqx0tR9HA'
        }

        response = requests.request("POST", self.voice_endpoint, headers=headers, data=payload)
        print(response.json())

        return response.json()

    def add_silence_buffer(self, aud_path, duration = 2):
        sec_seg = AudioSegment.silent(duration=duration*1000)
        song = AudioSegment.from_wav(aud_path)
        final = sec_seg + song + sec_seg
        final.export(aud_path, format="wav")

    def download_audio(self, endpoint, save_path):
        with open(save_path, 'wb') as f:
            f.write(requests.get(endpoint).content)


if __name__ == '__main__':
##
    base_path = './polycab/mul3'
    base_audio = os.listdir(base_path)
    base_audio = [x for x in base_audio if x.endswith('.wav')]


    # base_audio = gen_audio("Dear Ashu Jee") #narkeet call

    conv_path = 'polycab/res5'
    if not os.path.exists(conv_path):
        os.makedirs(conv_path)

    for i in tqdm(base_audio):

        subprocess.call(['svc', 'infer', '-m', 'polycab/G_1000.pth' ,'-c' ,'polycab/config.json', base_path + '/' + i, '-o',conv_path + '/' + i])
        # break
