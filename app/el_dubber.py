import json
import sys
import os

from src.eleven_labs_api import ElevenAPI
import  time
import shutil
from pathlib import Path
from src.utils import AwsBackNFro, merge_audio_video, mongo_db_connection
import ffmpeg
from src.speech2text import Speech2textSegment
from src.audio_processor import PrepareDataset, get_audio_len, audio_speeder, audio_timeline_swap, write_audio, vocals_seprator, add_instuments
from src.translation import GoogelTranslation
import librosa
from gtts import gTTS
import requests
from datetime import datetime
import uuid
from src.temp_infer_vc import infer_vm


BASE_DATA_FOLDER = os.path.join(Path(os.getcwd()).parent.absolute(),'data')

def download_file(url, save_path, chunk_size=1024):
                r = requests.get(url, stream=True)
                with open(save_path, 'wb') as fd:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)

def init_stats_log(mongo_db_object, uinq_id, total_progress):
    collection = 'status_url'
    status_json = {uinq_id: {'status':0,'current_progress':0 ,'total_progress': total_progress, }}
    
    mongo_id = mongo_db_object.insert_one(collection, status_json)
    return mongo_id

def update_log(mongo_db_object, mongo_id, uinq_id, status, current_progress, total_progress, s3_link=None):
    collection = 'status_url'

    status_json = {}
    status_json[uinq_id] = {'status':status}
    status_json[uinq_id]['current_progress'] = current_progress
    status_json[uinq_id]['total_progress'] = total_progress
    if s3_link:
        status_json[uinq_id]['s3_link'] = s3_link
    mongo_db_object.update_by_mongo_id(collection, mongo_id, status_json)


def el_dub_now(user_id,
               project_id,
               to_lang,
               source_video_url,
               uinq_id,
               mongo_db_object=None):

    try:

        BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id, "EL")
        
        # cleaning project folder
        if os.path.exists(BASE_PROJECT_FOLDER):
            shutil.rmtree(BASE_PROJECT_FOLDER)

        BASE_VIDEO_FOLDER = os.path.join(BASE_PROJECT_FOLDER,'BaseVideo')
        if not os.path.exists(BASE_VIDEO_FOLDER):
            os.makedirs(BASE_VIDEO_FOLDER)

        BASE_AUDIO_FOLDER = os.path.join(BASE_PROJECT_FOLDER,'BaseAudio')
        if not os.path.exists(BASE_AUDIO_FOLDER):
            os.makedirs(BASE_AUDIO_FOLDER)

        ## logging
        total_progress_log = 5
        mongo_id = init_stats_log(mongo_db_object, uinq_id=uinq_id, total_progress=total_progress_log)

        # if not os.path.exists(os.path.join(BASE_PROJECT_FOLDER, to_lang)):
        #     os.makedirs(os.path.join(BASE_PROJECT_FOLDER, to_lang))
            
        # log_json_file = os.path.join(BASE_PROJECT_FOLDER, to_lang,'log.json')
        
        # if os.path.exists(log_json_file):
        #     os.remove(log_json_file)
        # if not os.path.exists(log_json_file):
        #     with open(log_json_file, 'w') as f:
        #         dct = {'status':0,
        #         'current_progress': 0,
        #         'total_progress': total_progress_log}
        #         json.dump(dct, f)

        print("Downloading video")
        
        
        src_vid_extention = source_video_url.split('.')[-1]
        save_video_path = f'{BASE_VIDEO_FOLDER}/{user_id}_{project_id}.{src_vid_extention}'
        download_file(source_video_url, save_video_path)

        
        if not save_video_path.endswith('.mp4'):
            ## ffmpeg convert video (any) to video (mp4)
            f_name = ''.join(save_video_path.split('.')[:-1]) + '.mp4'
            save_path_out = "{}".format(f_name)
            print("savePath",save_path_out)

            (
                ffmpeg
                .input(save_video_path)
                .output(save_path_out)
                .run(overwrite_output=True)
                
            )
        ## ffmpeg convert video to audio
        save_path_out = f'{BASE_AUDIO_FOLDER}/{user_id}_{project_id}.wav'

        (
            ffmpeg
            .input(save_video_path)
            .output(save_path_out)
            .run(overwrite_output=True)
            
        )
        #update log - audio and video seperated
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                status=0,
                current_progress=1,
                total_progress=total_progress_log)

        
        # # print(user_id)

        main_audio = os.path.join(BASE_AUDIO_FOLDER,os.listdir(BASE_AUDIO_FOLDER)[0]) # audio to used
        print(main_audio)
        BASE_DEMUCS_OUT = os.path.join(BASE_PROJECT_FOLDER,"demucs_out")

        vocals_seprator(main_audio,BASE_DEMUCS_OUT)

        dmucs_folder = os.listdir(BASE_AUDIO_FOLDER)[0].split('.')[0]
        BASE_DEMUCS_OUT = os.path.join(BASE_DEMUCS_OUT, "htdemucs", dmucs_folder)
        main_audio = os.path.join(BASE_DEMUCS_OUT, "vocals.wav")


        # models and classes - start
        s2t = Speech2textSegment(batch_size=16,
                                    compute_type="float16",
                                    whisper_model="large-v2")
        prep_ds = PrepareDataset()
        api11 = ElevenAPI()

        # models and classes - close

        transcribe = s2t.transcribe_audio(main_audio)
        align_text = s2t.align_whisper_output(main_audio, transcribe)

        #update log - transcription and alignment done
        #update log - audio and video seperated
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                status=0,
                current_progress=2,
                total_progress=total_progress_log)

        #imp
        spk_diaz_dict = s2t.speaker_diarize(align_result=align_text, audio=main_audio) # speaker_sepration(text_alignment(transcribe))

        # base speaker_wise_audio
        BASE_SPEKAER_WISE_FOLDER = os.path.join(BASE_PROJECT_FOLDER,'BaseSpkrWiseFolder')
        if not os.path.exists(BASE_SPEKAER_WISE_FOLDER):
            os.makedirs(BASE_SPEKAER_WISE_FOLDER)

        print("making ds")
        prep_ds.make_dataset(BASE_SPEKAER_WISE_FOLDER, main_audio, spk_diaz_dict)


        # #adding speaker to elevenlabs
        speaker_names = {}

        for spk in os.listdir(BASE_SPEKAER_WISE_FOLDER):
            spk_name = "AI_DUB_"+user_id+'_'+project_id+'_'+spk
            voice_id = ''
            voice_id = api11.get_voice_id(spk_name)
            if len(voice_id) == 0:

                src_voice_list = []
                for f in os.listdir(os.path.join(BASE_SPEKAER_WISE_FOLDER, spk)):
                    if f.endswith('.wav'):
                        src_voice_list.append(os.path.join(BASE_SPEKAER_WISE_FOLDER, spk,f))

                voice_id = api11.add_voice(name=spk_name, src_voice_list=src_voice_list)
                print(voice_id)
                voice_id = voice_id['voice_id']
                speaker_names[spk] = voice_id
            else:
                voice_id = api11.get_voice_id(spk_name)
                speaker_names[spk] = voice_id
        
        print(speaker_names)

        
        # segments only for translation
        spk_diaz_dict_segments = spk_diaz_dict['segments']

        sdd_distil = {} # spk_diaz_dict distil
        for i, d in enumerate(spk_diaz_dict_segments):
            sdd_distil[i] = {
                "start": d['start'],
                "end": d['end'],
                "text": d["text"].strip(),
                "speaker": d['speaker']
            }
            if i-1 >= 0:
                sdd_distil[i-1]['gap'] = round(d['start']-sdd_distil[i-1]['end'], 2)

        with open(os.path.join(BASE_PROJECT_FOLDER,'tmp_seg_'+to_lang+".json"), 'w') as f:
            json.dump(sdd_distil, f)

        # translation
        trans = GoogelTranslation()
        translated_dict = trans.batch_translate(sdd_distil, to_lang)
        
        trans.batch_save_as_srt(translated_dict, to_lang, BASE_PROJECT_FOLDER)

        with open(os.path.join(BASE_PROJECT_FOLDER,'tmp_'+to_lang+".json"), 'w') as f:
            json.dump(translated_dict, f)

        #update log - translation done
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                    status=0,
                    current_progress=3,
                    total_progress=total_progress_log)
        
        # TTS part
        TTS_SEGMENT_TO_LANG_FOLDER = os.path.join(BASE_PROJECT_FOLDER, "TTS_SEGMENTS")
        if not os.path.exists(TTS_SEGMENT_TO_LANG_FOLDER):
            os.makedirs(TTS_SEGMENT_TO_LANG_FOLDER)
        
        TTS_SPEED_TO_LANG_FOLDER = os.path.join(BASE_PROJECT_FOLDER, "TTS_SPEED")
        if not os.path.exists(TTS_SPEED_TO_LANG_FOLDER):
            os.makedirs(TTS_SPEED_TO_LANG_FOLDER)
            

        # loading main audio
        aud_y, aud_sr = librosa.load(main_audio)
        aud_y = librosa.to_mono(aud_y)

        for i in range(0, int(get_audio_len(main_audio)*aud_sr)):
            aud_y[i] = 0

        for i, v in translated_dict.items():
            spkr = v['speaker']
            SPKR_TTS_SEGMENT_TO_LANG_FOLDER = os.path.join(TTS_SEGMENT_TO_LANG_FOLDER,spkr)
            if not os.path.exists(SPKR_TTS_SEGMENT_TO_LANG_FOLDER):
                os.makedirs(SPKR_TTS_SEGMENT_TO_LANG_FOLDER)

            save_file = os.path.join(SPKR_TTS_SEGMENT_TO_LANG_FOLDER, str(i)+".wav")
            

            # 11labs tts
            # voice_id = speaker_names[spkr]
            # api11.tts(voice_id=voice_id, text=v[to_lang], save_path=save_file) # v[to_lang] contains text
            # print(f"Save File: {save_file} | Voice ID: {voice_id}")


            # gtts
            tts = gTTS(text=v[to_lang], lang=to_lang)
            tts.save(save_file)

            ## so vits tts
            # resp_json = infer_vm(v[to_lang], "mr-IN")
            # shutil.move(resp_json['file_path'], save_file)

            ## auido speed up
            start = v['start']
            end = v['end']
            if i == len(translated_dict) - 1:
                gap = 0
            else:
                gap = v['gap']
            src_aud_len = end-start
            save_to = os.path.join(TTS_SPEED_TO_LANG_FOLDER, str(i)+".wav")
            audio_speeder(src_aud_len, save_file, save_to, aud_sr, gap)

            ## audio swap in main audio
            audio_timeline_swap(start, end, save_to, aud_y, aud_sr)

        
        ## saving result audio
        BASE_TRANSLATED_AUDIO = os.path.join(BASE_PROJECT_FOLDER, "result.wav")
        write_audio(audio_y = aud_y,
                    sr=22050,
                    path = BASE_TRANSLATED_AUDIO)
        
        # add bass, drums, others to audio
        bass_path = os.path.join(BASE_DEMUCS_OUT, 'bass.wav')
        drum_path = os.path.join(BASE_DEMUCS_OUT, 'drums.wav')
        other_path = os.path.join(BASE_DEMUCS_OUT, 'other.wav')


        RESULT_WITH_BGM = os.path.join(BASE_PROJECT_FOLDER, "result_bgm.wav")
        add_instuments(bass=bass_path,
                        drum=drum_path,
                        other=other_path,
                        vocal=BASE_TRANSLATED_AUDIO,
                        result_path=RESULT_WITH_BGM)

        #update log - adding instruments done
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                    status=0,
                    current_progress=4,
                    total_progress=total_progress_log)
        

        ## adding audio to video
        uuid_key = str(uuid.uuid4())[:8]
        res_name = uuid_key+"_"+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+f"_{user_id}_{project_id}.mp4"
        VIDEO_AUDIO_RES = os.path.join(BASE_PROJECT_FOLDER, res_name)
        merge_audio_video(RESULT_WITH_BGM, save_video_path, VIDEO_AUDIO_RES)


        # server_file = "http://54.197.1.158:8000"+RESULT_WITH_BGM.split("data")[1]
        ## uploading to s3
        aws = AwsBackNFro()
        with open(VIDEO_AUDIO_RES, 'rb') as f:
            server_file = aws.upload(f, VIDEO_AUDIO_RES.split('/')[-1])
        print(server_file)

        #update log - uploading to s3 done
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                    status=1,
                    current_progress=5,
                    total_progress=total_progress_log,
                    s3_link=server_file)

        ## add result to mongodb
        print("[+] Adding result to mongodb")
        mongo_result = mongo_db_connection('effy-ai-dub')
        collection = 'result_url'
        unique_user_id = user_id+'_'+project_id

        user_mongo_onj = mongo_result.find_one_by_uiqu_id(collection, unique_user_id)
        if user_mongo_onj:
            user_mongo_onj[unique_user_id].append(server_file)
            mongo_result.update_by_mongo_id(collection, user_mongo_onj['_id'], user_mongo_onj)
        else:
            user_mongo_onj = {unique_user_id: [server_file]}
            mongo_result.insert_one(collection, user_mongo_onj)
        print("[+] Done adding result to mongodb")

        ## delete voice id from 11labs if used any
        for spk, voice_id in speaker_names.items():
            delete_resp_obj = api11.delete_voice(voice_id)
            if delete_resp_obj.status_code == 200:
                print(f"Deleted {spk} from 11labs")
            else:
                print(f"Error deleting {spk} from 11labs")
    except Exception as e:
        print("Error Caught:",e)
        #update log - error
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                    status=-1,
                    current_progress=0,
                    total_progress=total_progress_log)