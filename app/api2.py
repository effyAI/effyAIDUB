from flask import Flask, request
from flask_restful import Resource, Api
from flask_autoindex import AutoIndex
import json
from sys import platform
import os
from so_vits_svc_fork.inference.main import infer


from src.eleven_labs_api import ElevenAPI
from src.train_clone_voice import Audio_split, TestMain
from src.base_tts_and_infer import AudioAPI
import  time
import torch
import shutil
from pathlib import Path
from src.utils import AwsBackNFro, AddEmphasis, merge_audio_video, mongo_db_connection
import librosa
from datetime import datetime
import uuid
from threading import Thread
from el_dubber import el_dub_now
from vm_dubber import vm_dub_now

app = Flask(__name__)
api = Api(app)


BASE_DATA_FOLDER = os.path.join(Path(os.getcwd()).parent.absolute(),'data')

status_url_table = mongo_db_connection('effy-ai-dub')

remove_file_list = ['configs', 'dataset', 'dataset_raw', 'filelists', 'logs', 'res', 'VoiceToClone']

def get_optimal_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class CloneAudioELabs(Resource):
    def post(self):
        try:
            req_json = json.loads(request.data.decode('utf-8'))
            print(req_json)
            if 'user_id' not in req_json or \
                'project_id' not in req_json or \
                'source_video_url' not in req_json or \
                'to_lang' not in req_json:
                return ({"Error": "Parameters Missing in Request"}, 400)
    
            print("testing")
            
            user_id = str(req_json['user_id'])
            project_id = str(req_json['project_id'])
            to_lang = req_json['to_lang']
            source_video_url = req_json['source_video_url']
            with open('lang_list.json','r') as f:
                lang_list = json.load(f)
            
            # if to_lang not in lang_list['tts_lang_ends']['elevenlabs_lang']:
            #     return ({"Error": "Pass Correct Language"}, 400)

            #status_url
            uinq_id = str(uuid.uuid4())
            
            Thread(target=el_dub_now, args=(user_id,
                                            project_id,
                                            to_lang,
                                            source_video_url, 
                                            uinq_id, 
                                            status_url_table,)).start()

            return({
                "status": "success",
                "description": "video processing started",
                "status_id": uinq_id
            }, 200)
            # return({
            #     "Vocal Path": BASE_TRANSLATED_AUDIO,
            #     "Result_WIth_BGM": RESULT_WITH_BGM
            # })
        except Exception as e:
            return ({"Error": str(e)})
        
    def get(self):
        print(request.data)

class ProgressStatusELabs_ID(Resource):
    def get(self):
        req_json = json.loads(request.data.decode('utf-8'))
        print(req_json)
        if 'status_id' not in req_json:
            return ({"Error": "Parameters Missing in Request"}, 400)

        status_id = str(req_json['status_id'])
        collection_name = 'status_url'
        status_obj = status_url_table.find_one_by_uiqu_id(collection_name, status_id)

        if status_obj is None:
            return ({"Error": "No Log Found for this ID"}, 400)
        
        ## delete if s3 link is present
        if 's3_link' in status_obj[status_id]:
            progress_status = status_obj[status_id] 
            mongo_id = status_obj['_id']
            status_url_table.delete_item(collection_name, {"_id": mongo_id})
            return progress_status

        return status_obj[status_id] 

class CloneAudioVM(Resource):
    def post(self):
        try:
            # print(request.json)
            req_json = json.loads(request.data.decode('utf-8'))
            # print(request.files['source_wav_file'].filename)
            if 'user_id' not in req_json or \
                'project_id' not in req_json or \
                'source_video_url' not in req_json or \
                'to_lang' not in req_json:
                return ({"Error": "Parameters Missing in Request"}, 400)
            
            user_id = req_json['user_id']
            project_id = req_json['project_id']
            to_lang = req_json['to_lang']
            source_video_url = req_json['source_video_url']
            with open('lang_list.json','r') as f:
                lang_list = json.load(f)

            if to_lang not in lang_list['tts_lang_ends']['voice_maker_lang']:
                return ({"Error": "Pass Correct Language"}, 400)

            #status_url
            uinq_id = str(uuid.uuid4())

            print("testing")
            Thread(target=vm_dub_now, args=(user_id,
                                            project_id,
                                            to_lang,
                                            source_video_url, 
                                            uinq_id, 
                                            status_url_table,)).start()

            # g_path = os.getcwd()+'/logs/44k/G_{}.pth'.format(EPOCH)
            # c_path = os.getcwd()+'/logs/44k/config.json'
            return({
                "status": "success",
                "description": "video processing started",
                "status_id": uinq_id
            }, 200)
        except Exception as e:
            return ({"Error Ha Ha Ha": str(e)})
        
    def get(self):
        print(request.data)

# class InferAudioELabs(Resource):
#     def post(self):
        
#         try:
#             req_data = json.loads(request.data.decode('utf-8'))

#             if 'user_id' not in req_data or \
#                 'project_id' not in req_data or \
#                 'text' not in req_data:
#                 return ({"Error": "Parameters Missing in Request"}, 400)

#             text = req_data['text']
#             user_id = req_data['user_id']
#             project_id = req_data['project_id']

#             if len(user_id) == 0:
#                 return ({"Error": "Please Provide User ID"}, 400)
#             if len(project_id) == 0:
#                 return ({"Error": "Please Provide Project ID"}, 400)
#             if len(text) == 0:
#                 return ({"Error": "Please Provide Text"}, 400)

#             BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id, "EL")

#             spk_name = user_id+'_'+project_id

#             conv_path = os.path.join(BASE_PROJECT_FOLDER,'res')
#             if not os.path.exists(conv_path):
#                 os.makedirs(conv_path)

#             res_file = os.path.join(conv_path,'test_{}.wav'.format(time.strftime("%d_%m_%Y_%H_%M_%S")))
            
#             api11 = ElevenAPI()
#             voice_id = api11.get_voice_id(spk_name)
#             api11.tts(voice_id=voice_id, text=text, save_path=res_file)
            
#             res_file = "http://34.239.163.60:8009"+res_file.split('data')[-1]
#             return ({"file_path": res_file}, 200)

#         except Exception as e:
#             return ({"Error": str(e)})
        
class InferAudioVM(Resource):
    def post(self):
        try:
            req_data = json.loads(request.data.decode('utf-8'))

            if 'user_id' not in req_data or \
                'project_id' not in req_data or \
                'text' not in req_data or \
                'language' not in req_data or \
                'gender' not in req_data:
                return ({"Error": "Parameters Missing in Request"}, 400)

            text = req_data['text']
            user_id = req_data['user_id']
            project_id = req_data['project_id']
            lang = str(req_data['language'])
            gender = str(req_data['gender'])

            if len(user_id) == 0:
                return ({"Error": "Please Provide User ID"}, 400)
            if len(project_id) == 0:
                return ({"Error": "Please Provide Project ID"}, 400)
            if len(text) == 0:
                return ({"Error": "Please Provide Text"}, 400)
            if len(lang) == 0:
                return ({"Error": "Please Provide Language"}, 400)
            if len(gender) == 0:
                return ({"Error": "Please Provide Gender"}, 400)

            with open('lang_list.json') as f:
                lang_list = json.load(f)
                if lang not in lang_list["tts_lang_ends"]["voice_maker_lang"]:
                    return ({"Error": "Language not supported"}, 400)
                voice_id = lang_list["tts_lang_ends"]["voice_maker_lang"][lang][gender]

            BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id, "VM")

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
            # res_file = "http://54.157.233.246:8000"+res_file.split('data')[-1]

            return ({"file_path": res_file}, 200)
        
        except Exception as e:
            return ({"Error": str(e)})
        
class ProgressStats(Resource):
    def get(self):    
        req_data = json.loads(request.data.decode('utf-8'))

        if 'user_id' not in req_data or \
            'project_id' not in req_data:
            return ({"Error": "Parameters Missing in Request"}, 400)

        user_id = str(req_data['user_id'])
        project_id = str(req_data['project_id'])

        
        if len(user_id) == 0:
            return ({"Error": "Please Provide User ID"}, 400)
        if len(project_id) == 0:
            return ({"Error": "Please Provide Project ID"}, 400)

        BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id, "VM")

        try:
            prog_file = os.path.join(BASE_PROJECT_FOLDER, 'train_progress.json') 
            with open(prog_file) as f:
                a = json.load(f)
                if a['Progress'] == 99:
                    a['Progress']+=1
                return a
        except FileNotFoundError as e:
            return {"Error": f"Wrong user/project id"}

class ProgressStatusVM(Resource):
    def get(self):
        req_json = json.loads(request.data.decode('utf-8'))
        print(req_json)
        if 'status_id' not in req_json:
            return ({"Error": "Parameters Missing in Request"}, 400)

        status_id = str(req_json['status_id'])
        collection_name = 'status_url'
        status_obj = status_url_table.find_one_by_uiqu_id(collection_name, status_id)

        if status_obj is None:
            return ({"Error": "No Log Found for this ID"}, 400)
        
        # # get train status from local file
        # if 'train_status_path' in status_obj[status_id]:
        #     with open(status_obj[status_id]['train_status_path']) as f:
        #         a = json.load(f)
        #         if a['Progress'] == 99:
        #             a['Progress']+=1
        
        ## delete if s3 link is present
        if 's3_link' in status_obj[status_id]:
            progress_status = status_obj[status_id] 
            mongo_id = status_obj['_id']
            status_url_table.delete_item(collection_name, {"_id": mongo_id})
            return progress_status

        return status_obj[status_id]

class GetLangList(Resource):
    def get(self):
        with open('lang_list.json') as f:
            lang_list = json.load(f)
            return lang_list

class GetLangListELabs(Resource):
    def get(self):
        with open('lang_list.json') as f:
            lang_list = json.load(f)
        return lang_list['tts_lang_ends']['elevenlabs_lang']

class GetLangListVM(Resource):
    def get(self):
        with open('lang_list.json') as f:
            lang_list = json.load(f)
        ret_data = {}
        for k,v in lang_list['tts_lang_ends']['voice_maker_lang'].items():
            ret_data[k] = v['lang']
        return ret_data
            
ppath = BASE_DATA_FOLDER # update your own parent directory here
AutoIndex(app, browse_root=ppath)

# api.add_resource(InferAudioELabs, '/inferE')
api.add_resource(InferAudioVM, '/inferV')
api.add_resource(CloneAudioELabs, '/cloneE')
api.add_resource(CloneAudioVM, '/cloneV')
api.add_resource(ProgressStats, '/progress')
# api.add_resource(ProgressStatusELabs, '/progressE')
api.add_resource(ProgressStatusELabs_ID, '/progressE')
api.add_resource(ProgressStatusVM, '/progressV')
api.add_resource(GetLangList, '/lang_list')
api.add_resource(GetLangListELabs, '/lang_list_E')
api.add_resource(GetLangListVM, '/lang_list_V')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)