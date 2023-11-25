from googletrans import Translator
import os

class GoogelTranslation:
    def __init__(self) -> None:
        self.translator = Translator()
    
    def batch_translate(self, sdd_distil, to_lang):
        """
        [0] : {
            "start": d['start'],
            "end": d['end'],
            "text": d["text"],
            "speaker": d['speaker']
        }
        """

        for k,v in sdd_distil.items():
            src_text = v['text']
            
            sdd_distil[k][to_lang] = self.translator.translate(src_text, dest=to_lang).text
        
        return sdd_distil
    
    def batch_save_as_srt(self, sdd_distil, to_lang, save_path):
        """
        [0] : {
            "start": d['start'],
            "end": d['end'],
            "text": d["text"],
            "speaker": d['speaker']
            to_lang: translated_text
        }
        """
        res = ""

        orignal_save = os.path.join(save_path, f'original_translated_{to_lang}.txt')
        # saving original text
        # 
        # idx [1 to end]
        # start --> end
        # text

        for k,v in sdd_distil.items():
            res += f'{k+1}\n'
            res += f'{v["start"]} --> {v["end"]}\n'
            res += f'Orignal Text - {v["text"]}\n'
            res += f'Translated Text - {v[to_lang]}\n\n'
        
        with open(orignal_save, 'w') as f:
            f.write(res)

