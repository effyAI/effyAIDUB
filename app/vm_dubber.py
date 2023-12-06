import os
from pathlib import Path
import shutil
import ffmpeg
import requests
from src.speech2text import Speech2textSegment
from src.audio_processor import PrepareDataset, get_audio_len, audio_speeder, audio_timeline_swap, write_audio, vocals_seprator, add_instuments, gender_detection
from src.infer_vm_sovits import get_audio_vm_so_vits
from src.train_clone_voice import TestMain
from src.translation import GoogelTranslation
import librosa
import json
import datetime
import uuid
from src.utils import AwsBackNFro, merge_audio_video, mongo_db_connection


BASE_DATA_FOLDER = os.path.join(Path(os.getcwd()).parent.absolute(),'data')
remove_file_list = ['configs', 'dataset', 'dataset_raw', 'filelists', 'logs', 'res', 'VoiceToClone']

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

def update_log(mongo_db_object, mongo_id, uinq_id, status, current_progress, total_progress, s3_link=None, train_status_path=None):
    collection = 'status_url'

    status_json = {}
    status_json[uinq_id] = {'status':status}
    status_json[uinq_id]['current_progress'] = current_progress
    status_json[uinq_id]['total_progress'] = total_progress
    # if train_status_path:
    #     status_json[uinq_id]['train_status_path'] = train_status_path
    if s3_link:
        status_json[uinq_id]['s3_link'] = s3_link
    mongo_db_object.update_by_mongo_id(collection, mongo_id, status_json)

def vm_dub_now(user_id,
               project_id,
               to_lang,
               source_video_url,
               uinq_id,
               mongo_db_object=None):
    
    BASE_PROJECT_FOLDER = os.path.join(BASE_DATA_FOLDER,user_id,project_id, "VM")
    
    try:
        
        if os.path.exists(BASE_PROJECT_FOLDER):
             shutil.rmtree(BASE_PROJECT_FOLDER)

        for rmf in remove_file_list:
                # print(rmf, os.path.join(BASE_PROJECT_FOLDER,rmf))
                if os.path.exists(os.path.join(BASE_PROJECT_FOLDER,rmf)):
                    print('[+] Cleaning - {}'.format(os.path.join(BASE_PROJECT_FOLDER,rmf)))
                    if rmf.endswith('.json'):
                        os.remove(os.path.join(BASE_PROJECT_FOLDER,rmf))
                    else:
                        shutil.rmtree(os.path.join(BASE_PROJECT_FOLDER,rmf))
        
        BASE_VIDEO_FOLDER = os.path.join(BASE_PROJECT_FOLDER,'BaseVideo')
        if not os.path.exists(BASE_VIDEO_FOLDER):
            os.makedirs(BASE_VIDEO_FOLDER)

        BASE_AUDIO_FOLDER = os.path.join(BASE_PROJECT_FOLDER,'BaseAudio')
        if not os.path.exists(BASE_AUDIO_FOLDER):
            os.makedirs(BASE_AUDIO_FOLDER)

        ## logging
        total_progress_log = 8
        mongo_id = init_stats_log(mongo_db_object, uinq_id=uinq_id, total_progress=total_progress_log)

        ## Downloading video

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

        main_audio = os.path.join(BASE_AUDIO_FOLDER,os.listdir(BASE_AUDIO_FOLDER)[0]) # audio to used
        print(main_audio)
        BASE_DEMUCS_OUT = os.path.join(BASE_PROJECT_FOLDER,"demucs_out")

        vocals_seprator(main_audio,BASE_DEMUCS_OUT)

        dmucs_folder = os.listdir(BASE_AUDIO_FOLDER)[0].split('.')[0]
        BASE_DEMUCS_OUT = os.path.join(BASE_DEMUCS_OUT, "htdemucs", dmucs_folder)


        # models and classes - start
        s2t = Speech2textSegment(batch_size=16,
                                    compute_type="float16",
                                    whisper_model="large-v2")
        prep_ds = PrepareDataset()
        # models and classes - close

        transcribe = s2t.transcribe_audio(main_audio)
        main_audio = os.path.join(BASE_DEMUCS_OUT, "vocals.wav")
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

        # making ds for traning
        prep_ds.make_dataset(BASE_SPEKAER_WISE_FOLDER, main_audio, spk_diaz_dict)

        ## identifying gender by voice
        # {speaker: gender}
        gender_dict = {}
        for spk in os.listdir(BASE_SPEKAER_WISE_FOLDER):
            gender_dict[spk] = gender_detection(os.path.join(BASE_SPEKAER_WISE_FOLDER,spk), spk)

        # segments only for translation
        spk_diaz_dict_segments = spk_diaz_dict['segments']

        sdd_distil = {} # spk_diaz_dict distil
        for i, d in enumerate(spk_diaz_dict_segments):

            sdd_distil[i] = {
                "start": d['start'],
                "end": d['end'],
                "text": d["text"].strip(),
                "speaker": d['speaker'],
                "gender" : gender_dict[d['speaker']] # adding gender
            }
            if i-1 >= 0:
                sdd_distil[i-1]['gap'] = round(d['start']-sdd_distil[i-1]['end'], 2)

        with open(os.path.join(BASE_PROJECT_FOLDER,'tmp_seg_'+to_lang+".json"), 'w') as f:
            json.dump(sdd_distil, f, indent=4)

        # translation
        trans = GoogelTranslation()
        translated_dict = trans.batch_translate(sdd_distil, to_lang)
        
        trans.batch_save_as_srt(translated_dict, to_lang, BASE_PROJECT_FOLDER)

        print("Translation Done")
        with open(os.path.join(BASE_PROJECT_FOLDER,'tmp_'+to_lang+".json"), 'w') as f:
            json.dump(translated_dict, f, indent=4)

        #update log - translation done
        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                    status=0,
                    current_progress=3,
                    total_progress=total_progress_log)
        

        ## trainig clone voice

        VOICE_TRAIN_PATH = os.path.join(BASE_PROJECT_FOLDER, "VoiceTrain")
        if not os.path.exists(VOICE_TRAIN_PATH):
            os.makedirs(VOICE_TRAIN_PATH)
        
        ## moving speakerwise IDs to train folder & traning
        
        mod_conf_path = {}
        for spk in os.listdir(BASE_SPEKAER_WISE_FOLDER):
            BASE_SPEKAER_TRAIN_FOLDER = os.path.join(VOICE_TRAIN_PATH, spk)
            shutil.copytree(os.path.join(BASE_SPEKAER_WISE_FOLDER,spk), os.path.join(BASE_SPEKAER_TRAIN_FOLDER, 'dataset_raw/sp1'))

            print(BASE_SPEKAER_TRAIN_FOLDER)

            svc = TestMain()
            # svc.preprocess(base_path=BASE_PROJECT_FOLDER)
            EPOCH = 1000

            prog = {"init_progress":0,
                "current_progress":0,
                "total_progress":0}
            prog_file = os.path.join(BASE_SPEKAER_TRAIN_FOLDER,'train_progress.json') 
            with open(prog_file,'w') as f:
                json.dump(prog, fp=f)

            svc.preprocess_n_train(base_path=BASE_SPEKAER_TRAIN_FOLDER, epochs=EPOCH, eval_interval=500, prog_file=prog_file)
        
            g_path = Path(BASE_SPEKAER_TRAIN_FOLDER) / "logs" / "44k"
            if g_path.is_dir():
                g_path = list(
                    sorted(g_path.glob("G_*.pth"), key=lambda x: x.stat().st_mtime)
                )[-1]
            
            c_path = os.path.join(BASE_SPEKAER_TRAIN_FOLDER, 'logs/44k/config.json')

            mod_conf_path[spk] = {
                'model': g_path,
                'config': c_path
            }
        
        # print(mod_conf_path)

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

            # tts = gTTS(text=v[to_lang], lang=to_lang)
            # tts.save(save_file)

            # so vits tts
            resp_json = get_audio_vm_so_vits(BASE_PROJECT_FOLDER,
                                             v[to_lang.split('-')[0]], # eg:- gu-IN so 'gu' only
                                            to_lang,
                                            v['gender'],
                                            mod_conf_path[spkr]['model'],
                                            mod_conf_path[spkr]['config'])
            shutil.move(resp_json['file_path'], save_file)

            # auido speed up
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
        res_name = uuid_key+"_"+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+f"_{user_id}_{project_id}.mp4"
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
                    current_progress=8,
                    total_progress=total_progress_log,
                    s3_link=server_file)


        


    except Exception as e:
        # error log
        error_log = os.path.join(BASE_PROJECT_FOLDER, 'thread_error.txt')
        with open(error_log, 'a') as f:
                f.write(str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')) + '\n')
                f.write(str(e) + '\n' + '------------------------------------------------------------------')
                f.write('\n\n')

        update_log(mongo_db_object,
                mongo_id = mongo_id,
                uinq_id=uinq_id,
                    status=-1,
                    current_progress=0,
                    total_progress=total_progress_log)