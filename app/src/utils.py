import boto3
import os
from nltk import word_tokenize, pos_tag
import nltk
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from pymongo.mongo_client import MongoClient
import certifi

class AwsBackNFro():
    def __init__(self):
        self.aws_access_key_id = ""
        self.aws_secret_access_key=""
        self.region_name="us-east-1"

        self.s3 = boto3.client('s3', aws_access_key_id=self.aws_access_key_id,
                              aws_secret_access_key=self.aws_secret_access_key,
                              region_name=self.region_name)
        
        self.s3_bucket = boto3.resource('s3', aws_access_key_id=self.aws_access_key_id,
                          aws_secret_access_key=self.aws_secret_access_key,
                          region_name=self.region_name)

        self.bucket_name = 'effy-ai-dub-results'

    def upload(self, file_obj, file_path):
        self.s3.upload_fileobj(file_obj, self.bucket_name, file_path)
        s3_url = f'https://{self.bucket_name}.s3.amazonaws.com/{file_path}'
        return s3_url

    def download(self, file_path, local_file_path):
        self.s3.download_file(self.bucket_name, file_path, local_file_path)
        return local_file_path

    def upload_dict(self, file_dict):
        for subdir, dirs, files in os.walk(file_dict):
            for file in files:
                full_path = os.path.join(subdir, file)
                with open(full_path, 'rb') as data:
                    self.s3_bucket.Bucket(self.bucket_name).put_object(Key=full_path[len(file_dict)+1:], Body=data)

        print("Folders Uploaded to S3")

class AddEmphasis():
    def __init__(self):
        self.emphs = {
            'strong': ["<emphasis level='strong'>", "</emphasis>"]
        }

        self.tag_word = ['NN', 'NNS', 'NNPS', 'NNP']
    
    def install_nltk_req(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def extract_nouns(self, text, ):
        words = word_tokenize(text)
        tagged_words = pos_tag(words)

        nouns = [word for word, tag in tagged_words if tag in self.tag_word]
        return nouns

    def add_emp_strong(self,text, grammer):
        words = word_tokenize(text)
        res = []
        for i in words:
            if i in grammer:
                w = self.emphs['strong'][0] + i + self.emphs['strong'][1]
            else:
                w = i
            res.append(w)

        return (' '.join(res))

def merge_audio_video(audio_path, video_path, output_path):
    print(audio_path)
    print(video_path)
    print(output_path)
    audio = AudioFileClip(audio_path)
    video = VideoFileClip(video_path)
    final_audio = CompositeAudioClip([audio])
    final_video = video.without_audio()
    final_video.audio = final_audio
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, fps=30, preset='ultrafast', threads=8)

class mongo_db_connection():
    def __init__(self, db_name):
        self.uri = "mongodb+srv://effybizai:AhM2SPj8dKfLId89@cluster0.yfq6agh.mongodb.net/?retryWrites=true&w=majority"
        self.ca = certifi.where()
        self.client = MongoClient(self.uri, tlsCAFile=self.ca)
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

        self.db = self.client[db_name]

    def insert_one(self, collection_name, item):
        collection = self.db[collection_name]
        mongo_id = collection.insert_one(item)
        return mongo_id.inserted_id
    
    def get_all(self, collection_name):
        collection = self.db[collection_name]
        all_items = collection.find()
        return all_items

    def delete_item(self, collection_name, item):
        collection = self.db[collection_name]
        collection.delete_one(item)

    def find_one_by_uiqu_id(self, collection_name, id):
        all_items = self.get_all(collection_name)
        for item in all_items:
            if id in item:
                return item
        return None
    
    def update_by_mongo_id(self, collection_name, mongo_id, new_json):
        collection = self.db[collection_name]
        collection.update_one({"_id": mongo_id}, {"$set": new_json})
                
if __name__ == "__main__":
    merge_audio_video("/mnt/sd1/AI_DUB/data/52/uplr/EL/result_bgm.wav",
                      "/mnt/sd1/AI_DUB/data/52/uplr/EL/BaseVideo/52_uplr.mp4",
                      "/mnt/sd1/AI_DUB/app/src/test.mp4")