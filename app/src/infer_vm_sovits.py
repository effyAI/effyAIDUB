
from src.base_tts_and_infer import AudioAPI
import os
import time
from pathlib import Path
import json
from so_vits_svc_fork.inference.main import infer
import torch


LANG_LIST_PATH = os.path.join(Path(os.getcwd()).parent.absolute(),'lang_list.json')

def get_optimal_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_audio_vm_so_vits(BASE_PROJECT_FOLDER, text, lang, gender, g_path, c_path):

    audio = AudioAPI()

    # process text add empas
    # empas_obj = AddEmphasis()
    # grammer_noun = empas_obj.extract_nouns(text)
    # text = empas_obj.add_emp_strong(text, grammer_noun)

    print(text)
    with open('lang_list.json') as f:
        lang_list = json.load(f)
        voice_id = lang_list["tts_lang_ends"]["voice_maker_lang"][lang][gender]
    print(voice_id)
    # voice maker make audio
    vm_response = audio.get_audio_json(text, lang_id=lang, voice_id=voice_id)

    ## from voice_maker
    vm_path = os.path.join(BASE_PROJECT_FOLDER,'vm')
    if not os.path.exists(vm_path):
        os.makedirs(vm_path)

    vm_file = os.path.join(vm_path,'test_{}.wav'.format(time.strftime("%d_%m_%Y_%H_%M_%S")))

    if "path" in vm_response:
        audio.download_audio(vm_response['path'], vm_file) # downliading audio
        audio.add_silence_buffer(vm_file,  0) # adding silence buffer in start and end

    else:
        return {"Error": vm_response}
    conv_path = os.path.join(BASE_PROJECT_FOLDER,'res')
    if not os.path.exists(conv_path):
        os.makedirs(conv_path)

    res_file = os.path.join(conv_path,'test_{}.wav'.format(time.strftime("%d_%m_%Y_%H_%M_%S")))
    # subprocess.call(['svc', 'infer', '-m', g_path ,'-c' ,c_path, base_aud_path, '-o',res_file])
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

    return {"file_path": res_file}