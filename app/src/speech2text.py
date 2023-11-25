import whisperx
import gc
import torch
import time

class Speech2textSegment:
    def __init__(self,
                 batch_size, 
                 compute_type,
                 whisper_model = "large-v3") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.whisper_model_type = whisper_model
        self.hf_token = "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"
    
    def transcribe_audio(self, audio_path, deallocate=True):
        whsiper_model = whisperx.load_model(self.whisper_model_type,
                                         self.device,
                                         compute_type=self.compute_type)
        audio = whisperx.load_audio(audio_path)
        result = whsiper_model.transcribe(audio, batch_size=self.batch_size)

        if deallocate==True:
            self.model_dealocator(whsiper_model)

        return result
    
    def align_whisper_output(self, audio, segment_result, deallocate=True):
        model_a, metadata = whisperx.load_align_model(language_code=segment_result['language'], device=self.device)
        result_a = whisperx.align(segment_result['segments'], model_a, metadata, audio, self.device, return_char_alignments=False )
        
        if deallocate==True:
            self.model_dealocator(model_a)

        return result_a
    
    def speaker_diarize(self, audio, align_result, deallocate=True):
        """
        {
        "start": d['start'],
        "end": d['end'],
        "text": d["text"]
        "speaker" : d["speaker"]
        }
        """
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio)
        result_sd = whisperx.assign_word_speakers(diarize_segments, align_result)
        if deallocate==True:
            self.model_dealocator(diarize_model)
            
        return result_sd

    def model_dealocator(self, model):
        gc.collect()
        torch.cuda.empty_cache()
        del model
    
if __name__ == "__main__":
    a = Speech2textSegment(16, "float16", "large-v2")
    audio = "/home/infinity/Desktop/Effy_Internship/new_code_nov/ai_dub/tests/demoise_1.wav"
    transcried = a.transcribe_audio(audio)
    print(transcried)
    print("-----------------------------------")
    align_text = a.align_whisper_output(audio, transcried)
    print(align_text)
    print("------------------------------------")
    spk_dia_dict = a.speaker_diarize(audio, align_text)
    print(spk_dia_dict)


    
