from flask import Flask, request
from flask_restful import Resource, Api
import json
from sys import platform
import os
from so_vits_svc_fork.inference.main import infer

from src.train_clone_voice import Audio_split, TestMain
from src.base_tts_and_infer import AudioAPI
import  time
import torch
import shutil
from pathlib import Path
from threading import Thread
from src.utils import AwsBackNFro, AddEmphasis

app = Flask(__name__)
api = Api(app)


BASE_DATA_FOLDER = os.path.join(Path(os.getcwd()).parent.absolute(),'data')

remove_file_list = ['configs', 'dataset', 'dataset_raw', 'filelists', 'logs', 'res']

def get_optimal_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class CloneAudio(Resource):
    def post(self):
        # print(request.files['source_wav_file'].filename)
        if 'user_id' not in request.form:
            return ({"Error": "Please Provide User ID"}, 400)
        if 'project_id' not in request.form:
            return ({"Error": "Please Provide Project ID"}, 400)
        if 'source_wav_file' not in request.files:
            return ({"Error": "Please Provide Source wav file ID"}, 400)
        
        user_id = request.form['user_id']
        project_id = request.form['project_id']

        BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id)

        for rmf in remove_file_list:
            # print(rmf, os.path.join(BASE_PROJECT_FOLDER,rmf))
            if os.path.exists(os.path.join(BASE_PROJECT_FOLDER,rmf)):
                print('[+] Cleaning - {}'.format(os.path.join(BASE_PROJECT_FOLDER,rmf)))
                if rmf.endswith('.json'):
                    os.remove(os.path.join(BASE_PROJECT_FOLDER,rmf))
                else:
                    shutil.rmtree(os.path.join(BASE_PROJECT_FOLDER,rmf))
        

        BASE_CLONE_FOLDER = os.path.join(BASE_PROJECT_FOLDER,'VoiceToClone')
        if not os.path.exists(BASE_CLONE_FOLDER):
            os.makedirs(BASE_CLONE_FOLDER)

        save_path = "{}/Testaudio.wav".format(BASE_CLONE_FOLDER)
        request.files['source_wav_file'].save(save_path)
        
        print(user_id)
        
        ## spliting and training in on voice
        spl_au = Audio_split()

        resp = spl_au.splitter(audio_path=save_path, out_path=BASE_PROJECT_FOLDER, split_sec=8)
        if resp['status'] == 0:
            return (resp['reason'], 400) 
        if resp['status'] == 1:
            print("Traning Will start Shortly")

        svc = TestMain()
        # svc.preprocess(base_path=BASE_PROJECT_FOLDER)
        EPOCH = 1000

        #starting thread for training
        Thread(target=svc.preprocess_n_train, args=(BASE_PROJECT_FOLDER,EPOCH,500)).start()

        g_path = os.getcwd()+'/logs/44k/G_{}.pth'.format(EPOCH)
        c_path = os.getcwd()+'/logs/44k/config.json'
        return ({"Sucess": "Training Started!",
                 "user_id": user_id,
                 "project_id": project_id}, 200)

    def get(self):
        print(request.data)

class InferAudio(Resource):
    def post(self):
        req_data = json.loads(request.data.decode('utf-8'))

        text = req_data['text']
        user_id = req_data['user_id']
        project_id = req_data['project_id']
        lang = str(req_data['language'])
        voice_id = str(req_data['voice_id'])

        BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id)

        audio = AudioAPI()

        # process text add empas
        # empas_obj = AddEmphasis()
        # grammer_noun = empas_obj.extract_nouns(text)
        # text = empas_obj.add_emp_strong(text, grammer_noun)

        print(text)

        # voice maker make audio
        vm_response = audio.get_audio_json(text, lang_id=lang, voice_id=voice_id)

        ## from voice_maker
        vm_path = os.path.join(BASE_PROJECT_FOLDER,'vm')
        if not os.path.exists(vm_path):
            os.makedirs(vm_path)

        vm_file = os.path.join(vm_path,'test_{}.wav'.format(time.strftime("%d_%m_%Y_%H_%M_%S")))

        if "path" in vm_response:
            audio.download_audio(vm_response['path'], vm_file) # downliading audio
            audio.add_silence_buffer(vm_file,  1.2) # adding silence buffer in start and end

        else:
            return {"Error": vm_response}
        conv_path = os.path.join(BASE_PROJECT_FOLDER,'res')
        if not os.path.exists(conv_path):
            os.makedirs(conv_path)

        res_file = os.path.join(conv_path,'test_{}.wav'.format(time.strftime("%d_%m_%Y_%H_%M_%S")))
        # subprocess.call(['svc', 'infer', '-m', g_path ,'-c' ,c_path, base_aud_path, '-o',res_file])
        g_path = Path(BASE_PROJECT_FOLDER) / "logs" / "44k"
        if g_path.is_dir():
            g_path = list(
                sorted(g_path.glob("G_*.pth"), key=lambda x: x.stat().st_mtime)
            )[-1]
        c_path = Path(BASE_PROJECT_FOLDER) / "logs" / "44k" / "config.json"
        infer(
            # paths
            input_path=vm_file,
            output_path=res_file,
            model_path=g_path,
            config_path=c_path,
            recursive=False,
            # svc config
            speaker=str,
            cluster_model_path=None,
            transpose=0,
            auto_predict_f0=True,
            cluster_infer_ratio=0,
            noise_scale=0.4,
            f0_method="dio",
            # slice config
            db_thresh = -20,
            pad_seconds = 0.5,
            chunk_seconds = 0.5,
            absolute_thresh=False,
            max_chunk_seconds=40,
            device=get_optimal_device(),
        )

        return ({"file_path": res_file}, 200)

class ProgressStats(Resource):
    def get(self):

        req_data = json.loads(request.data.decode('utf-8'))
        user_id = str(req_data['user_id'])
        project_id = str(req_data['project_id'])

        BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id)

        try:
            prog_file = os.path.join(BASE_PROJECT_FOLDER, 'train_progress.json') 
            with open(prog_file) as f:
                a = json.load(f)
                # if a['Progress'] == 99:
                #     a['Progress']+=1
                return a
        except:
            return {"Progress":0}
            

api.add_resource(InferAudio, '/infer/')
api.add_resource(CloneAudio, '/clone/')
api.add_resource(ProgressStats, '/progress/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8009)