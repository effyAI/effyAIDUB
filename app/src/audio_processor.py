import os
import librosa
import soundfile as sf
from pydub import AudioSegment
from audiostretchy.stretch import stretch_audio
import math
import demucs.separate
from transformers import pipeline


class PrepareDataset:
    def __init__(self,
                 min_aud_len=5) -> None:
        self.min_aud_len = min_aud_len


    def make_dataset(self, base_speaker_save_folder, base_audio_path, spk_dia_dict):
        
        y, sr = librosa.load(base_audio_path, mono=True, sr =22050)

        file_index = {}

        segemt_list = spk_dia_dict["segments"]
        # print(segemt_list)
        for d in segemt_list:
            # print(d)
            start = d['start'],
            end = d['end'],
            text = d["text"]
            speaker = d["speaker"]

            if speaker not in file_index:
                file_index[speaker]=0
            # print(start, type(start), end, type(end), text, speaker)
            start = start[0]
            end = end[0]

            print(f'Start: {start} | end: {end} | fname: {file_index[speaker]} | totTime: {end-start}')
            # sperate_speaker_folder
            
            spkr_path = os.path.join(base_speaker_save_folder, speaker)
            if not os.path.exists(spkr_path):
                os.makedirs(spkr_path)
            
            if (end-start) > 5: # if segment is greater than 5 sec
                
                sf.write(spkr_path+'/'+str(file_index[speaker])+'.wav',
                         y[int(sr*start): int(sr*end)], 22050, 'PCM_24')
                if speaker in file_index:
                    file_index[speaker]+=1

def vocals_seprator(audio, save_path):
    demucs.separate.main([audio, "--out", save_path])

def get_audio_len(aud_path):
    """Get audio length in seconds"""
    audio = AudioSegment.from_file(aud_path)
    return audio.duration_seconds

def center_align(src_aud_len, saved_audio_path):
    """Add silence to start and end of audio file"""
    saved_audio = AudioSegment.from_file(saved_audio_path)
    saved_duration = get_audio_len(saved_audio_path)

    diff = round(src_aud_len - saved_duration, 2)
    print("diff:", diff, "saved_duration", saved_duration)
    silence_buffer = AudioSegment.silent(duration=(diff/2)*1000)
    final_audio = silence_buffer + saved_audio + silence_buffer
    final_audio.export(saved_audio_path, format="wav")

def audio_speeder(src_aud_len, targ_audio_file_path, save_to, base_aud_sr, gap):
    """Do in plae speed change of audio file"""

    tmp_data, tmp_sr = sf.read(targ_audio_file_path)
    sf.write(targ_audio_file_path, tmp_data, tmp_sr)
    # audio, sr = librosa.load(targ_audio_file_path, sr=base_aud_sr)
    
    # convert to 1 channel
    # audio = librosa.to_mono(audio)


    aud_len = round(get_audio_len(targ_audio_file_path), 5)
    # speed = round(aud_len/src_aud_len, 2)
    speed = round(aud_len/src_aud_len, 5)

    stretch = round(1/speed, 5) # because we are stretching audio / stretch ratio

    print("src len:",src_aud_len, "audio duration:",aud_len, "speed: ", speed, "stretch:", stretch)

    if stretch > 1.25: # very slow speaking resolve
        stretch = 1.25
        stretch_audio(targ_audio_file_path, save_to, ratio=stretch)
        center_align(src_aud_len, save_to)
        print("center aligning")

    elif stretch < 0.60: # very fast speaking resolve
        # recalculating speed, stretch using gap

        #using 80% of gap
        gap = gap*0.80

        new_src_aud_len = src_aud_len + gap
        speed = round(aud_len/new_src_aud_len, 5)
        stretch = round(1/speed, 5)
        if stretch > 1.25: # caping stretch to 1.25
            stretch = 1.25
        stretch_audio(targ_audio_file_path, save_to, ratio=stretch)

    else:

        # if speed > 0.5 and speed < 1.5:
        print("speeding up")
        stretch_audio(targ_audio_file_path,save_to , ratio=stretch)
            # fast_aud = librosa.effects.time_stretch(y =audio, rate=speed)
        # else:
        #     stretch = 0.60
        #     stretch_audio(targ_audio_file_path, save_to, ratio=stretch)
            # fast_aud = librosa.effects.time_stretch(y =audio, rate=speed)
        # save_file = os.path.join(os.getcwd() ,res_path, f"{k}.wav")
        # sf.write(save_file, fast_aud, sr,format='wav' )

    ## trim extra silence
    tmp_data, tmp_sr = librosa.load(save_to)
    tmp_data = librosa.to_mono(tmp_data)
    tmp_data = librosa.effects.trim(tmp_data)[0]
    sf.write(save_to, tmp_data, tmp_sr, 'PCM_24')
    print("Result AUdio Dur",get_audio_len(save_to))

def audio_timeline_swap(start, end, targ_aud_path, aud_y, aud_sr):
    """Swap audio timeline"""
    tgt_aud, tgt_sr = librosa.load(targ_aud_path, sr=aud_sr)
    tgt_aud = librosa.to_mono(tgt_aud)
    print("tgt sr:", tgt_sr)
    # sf.write("test.wav", aud_y[math.floor(start*tgt_sr):math.ceil(end*tgt_sr)], tgt_sr, 'PCM_24')
    
    for i in range(0, int(get_audio_len(targ_aud_path)*tgt_sr)):
        if tgt_aud[i] != 0 and len(aud_y) > math.floor(start)*tgt_sr+i:
            aud_y[math.floor(start*tgt_sr)+i] = tgt_aud[i]
            
def write_audio(audio_y, sr, path):
    sf.write(path, audio_y, sr, 'PCM_24')

def add_instuments(bass,drum, other, vocal, result_path):
    bass = AudioSegment.from_file(bass, format='wav')
    drum = AudioSegment.from_file(drum, format='wav')
    other = AudioSegment.from_file(other, format='wav')
    vocal = AudioSegment.from_file(vocal, format='wav')

    overlay = bass.overlay(drum, position=0)
    overlay = overlay.overlay(other, position=0)
    overlay = overlay.overlay(vocal, position=0)

    overlay.export(result_path, format="wav")


def gender_detection(parent_audio_path, speaker_name):

    data = {}
    data[speaker_name] = {"male":0, "female":0}
                          
    for a in os.listdir(parent_audio_path):
        pipe = pipeline("audio-classification", model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
        # res = [{'score': 0.9987540245056152, 'label': 'male'}, {'score': 0.001245927414856851, 'label': 'female'}]
        res = pipe(os.path.join(parent_audio_path, a))
        max_score = 0
        for r in res:
            if r['score'] > max_score:
                max_score = r['score']
                max_label = r['label']
        data[speaker_name][max_label]+=1
    
    # return max gender
    return max(data[speaker_name], key=data[speaker_name].get)
    