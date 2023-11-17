#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import av
import os
import re
import cv2
import torch
import json
import time
import librosa
import numpy as np
from glob2 import glob
from random import shuffle
from time import time
import warnings
from torchvggish import vggish
from transformers import AutoImageProcessor, TimesformerModel, TimesformerConfig
from transformers import BertModel,BertTokenizer
from timesformer.models.vit import TimeSformer
np.random.seed(0)
BERT_PATH = './pretrainedmodels/bert-base-uncased'
TimeSformer_PATH = './pretrainedmodels/TimeSformer/TimeSformer_divST_96x4_224_K600.pyth'
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings('ignore', message="Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.")
######################################################################################################## 
     
def check_file(fn):
    with open(fn, 'r') as f:
        words = [x.replace('"', '').replace(":", "").replace('.', '').replace(',', '').rstrip().split(' ')[0] for x in f.readlines()]
    return len(words) > 20

category15 = ['films','arts','comic','dance','fashion','music','game','handicraft','news','photography','publication','tech','theatre','food','design']
category10 = ['fashion','game','handicraft','news','photography','publication','tech','theatre','food','design']
category2 = ['comic','dance']

def get_dataset(kickstarter_root, glove_root):
    train_all = []
    val_all = []
    for c in category15:
        success_data_name = os.listdir(os.path.join(kickstarter_root , c, "successful"))
        success_data_name = [os.path.join(kickstarter_root,c, "successful", d) for d in success_data_name]
        success_data_name = [i for i in success_data_name if ('f1.txt' in os.listdir(i) and "f1.mp4" in os.listdir(i))]
        success_data_name = [i for i in success_data_name if check_file(os.path.join(i, 'f1.txt'))]
        success_labels = [1, ] * len(success_data_name)

        failed_data_name = os.listdir(os.path.join(kickstarter_root, c, "failed"))
        failed_data_name = [os.path.join(kickstarter_root,c, "failed", d) for d in failed_data_name]
        failed_data_name = [i for i in failed_data_name if ('f1.txt' in os.listdir(i) and "f1.mp4" in os.listdir(i))]
        failed_data_name = [i for i in failed_data_name if check_file(os.path.join(i, 'f1.txt'))]
        failed_labels = [0, ] * len(failed_data_name)

        data = success_data_name + failed_data_name
        labels = success_labels + failed_labels

        data_ = [[i, j] for i, j in zip(data, labels)]
        shuffle(data_)
        globals()[f'train_{c}']= data_[:int(len(data_) * 0.8)]
        globals()[f'val{c}'] = data_[int(len(data_) * 0.8):]
        #globals()[f'test{c}'] = data_[int(len(data_) * 0.8):]

        train_all.extend(globals()[f'train_{c}'])
        val_all.extend(globals()[f'val{c}'])


    train_dataset = K_dataset(train_all)
    val_dataset = K_dataset(val_all)

               
    return train_dataset, val_dataset



class K_dataset():
    def __init__(self, all_data):
        self.data = [i[0] for i in all_data]
        self.label = [i[1] for i in all_data]
        self.MAX_FRAME_NUMBER = 200
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.modelt = BertModel.from_pretrained(BERT_PATH)
        self.modelv = TimeSformer(img_size=224, num_classes=600, num_frames=32, attention_type='divided_space_time',pretrained_model=TimeSformer_PATH )
        self.modela = vggish().eval()
    
    def __len__(self):
        return len(self.data)   
    
    def __getitem__(self, item):
        video_name = os.path.join(self.data[item], 'f1.mp4')
        video_np_name = os.path.join(self.data[item], 'f1_timesformer_32.npy')
        vocab_file = os.path.join(self.data[item], 'f1.txt')
        vocab_np_file = os.path.join(self.data[item], 'f1_v_bert.npy')
        audio_name = video_name
        audio_np_name_vggish_128 = os.path.join(self.data[item], 'f1_a_vgg128_new.npy')
        meta_file = vocab_file
        meta_json_file = os.path.join(self.data[item], 'meta3.json')
        
        label = self.label[item]
        
        # Meta
        if os.path.exists(meta_json_file):
            with open(meta_json_file, 'r') as file:
                meta = json.load(file)
                #meta1 = data.get("meta1")
        else:
            with open(meta_file, 'r') as f:
                meta = f.readline().rstrip()
                start_index = meta.find("{")
                end_index = meta.find("}") + 1
                result = meta[start_index:end_index].replace("'", "\"")
                pattern = r'(\w+):"([^"]*)"'
                matches = re.findall(pattern, result)
                dictionary = {match[0]: match[1] for match in matches}
                meta = dictionary
                with open(meta_json_file, 'w') as file:
                    json.dump(dictionary, file)  
        
        #Text
        if os.path.exists(vocab_np_file):
            word_features = np.load(vocab_np_file)
        else:
            with open(vocab_file, 'r') as f:
                text = f.read().replace('"', ' ').replace(":", " ").replace('.', ' ').replace(',', ' ').rstrip()
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            

            with torch.no_grad():
                output = self.modelt(**inputs)
            word_features = output.last_hidden_state.mean(dim=1)  
            np.save(vocab_np_file, word_features)

                
        #Video
        
        if os.path.exists(video_np_name):
            outputs = np.load(video_np_name)
        else:
            flag=True
            outputs = []
            
            video = cv2.VideoCapture(video_name)
            fps = video.get(cv2.CAP_PROP_FPS)
            frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            try:
                frame_num = min(int(frames / fps), self.MAX_FRAME_NUMBER)
            except:
                frame_num=0
            
            if frames < 100:
                flag=False
            if flag==True:
                interval = int(frame_num / 32)  
                frames = []
                for i in range(32):
                    frame_indx = i * interval * fps
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_indx - 1)
                    ret, frame = video.read()
                    if ret:
                        H, W = frame.shape[:2]
                        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
                        if H > W:
                            pad_left = (H - W) // 2
                            pad_right = H - W - pad_left
                        if H < W:
                            pad_top = (W - H) // 2
                            pad_bottom = W - H - pad_top
                        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
                        frame = cv2.resize(frame, (224, 224))         
                        frames.append(frame)
                    else:
                        print(f"[ERROR] {video_name}: frames {frame_indx} cap failed!")         
                frames = np.stack(frames, 0)
                frames = torch.FloatTensor(frames)
                frames = frames.permute(3, 0, 1, 2)
                frames = torch.unsqueeze(frames, dim=0)
                
                model = self.modelv.eval()
                outputs = model(frames).detach().numpy()
                
                np.save(video_np_name,outputs)
               
        #Audio
        if os.path.exists(audio_np_name_vggish_128):
            vggish_out = np.load(audio_np_name_vggish_128)
        else:
            mfcc = extract_mfcc_features(audio_name,128)   

            embedding_model =  self.modela
            vggish_out = embedding_model.forward(mfcc).unsqueeze(0).detach().numpy()
            np.save(audio_np_name_vggish_128, vggish_out)           
       
        return meta, outputs, word_features,vggish_out,label

    

def extract_mfcc_features(audio_name,n_mfcc): 
    try:
        X, sample_rate = librosa.load(audio_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs = np.mean(mfccs, axis=1)
        resized_mfcc = cv2.resize(mfccs, (64, 96), interpolation=cv2.INTER_LINEAR)
        expanded_mfcc = np.expand_dims(resized_mfcc, axis=0)  # Add batch dimension
        expanded_mfcc = np.expand_dims(expanded_mfcc, axis=1)  # Add channel dimension - 1 channel (grayscale)
        expanded_mfcc = expanded_mfcc.astype(np.float32)
        mfccs = torch.tensor(expanded_mfcc)
        return mfccs   

    except :
        mfccs = np.zeros([n_mfcc,])
        resized_mfcc = cv2.resize(mfccs, (64, 96), interpolation=cv2.INTER_LINEAR)
        expanded_mfcc = np.expand_dims(resized_mfcc, axis=0)  # Add batch dimension
        expanded_mfcc = np.expand_dims(expanded_mfcc, axis=1)  # Add channel dimension - 1 channel (grayscale)
        expanded_mfcc = expanded_mfcc.astype(np.float32)
        mfccs = torch.tensor(expanded_mfcc)
        return mfccs
    
def duration_process(string):
    pattern = r'(\d{4})-(\d{2})-(\d{2})'  
    matches = re.findall(pattern, string)
    date_info1 = matches[0]  
    date_info2 = matches[1]

    year1 = date_info1[0]
    month1 = date_info1[1]
    day1 = date_info1[2]
    year2 = date_info2[0]
    month2 = date_info2[1]
    day2 = date_info2[2]

    start = year1 + month1 + day1
    end = year2 + month2 + day2
    return float(start),float(end)



    
def collate(batch):
    meta_indexs = []
    metas = []
    vec_indexs = []
    vecs = []
    frame_indexs = []
    frames = []
    audio_indexs = []
    audios = []
    labels = []
    pseudos = []


    for i, b in enumerate(batch):
        meta = b[0]
        frame = b[1]
        vec = b[2]
        audio = b[3]
        label = b[4]
        
        if len(frame) ==0 or len(meta)==0:
            return torch.ones([1]), torch.zeros([1]),torch.zeros([1]), torch.zeros([1]),torch.zeros([1]), torch.zeros([1]),torch.zeros([1]), torch.zeros([1]), torch.zeros([1])

        metas_ = []
        
        meta_finish = float(meta['meta1'])
        meta_target = float(meta['meta2'])
        duration = meta['meta4']

        try:
            meta_start,meta_end = duration_process(duration)
        except:
            return torch.ones([1]), torch.zeros([1]),torch.zeros([1]), torch.zeros([1]),torch.zeros([1]), torch.zeros([1]),torch.zeros([1]), torch.zeros([1]), torch.zeros([1])
        
        try:
            meta_create = float(meta['meta3'])
        except ValueError:
            meta_create = 0.0
            
        metas_=[meta_finish,meta_target,meta_start,meta_end,meta_create]
        
        meta_indexs += [i, ] * frame.shape[0]
        audio_indexs += [i, ] * audio.shape[0]
        frame_indexs += [i, ] * frame.shape[0]
        vec_indexs += [i, ] * vec.shape[0]
        
        metas.append(list([metas_]))
        audios.append(audio)
        frames.append(frame)
        vecs.append(vec)
        labels.append(label)

        
    metas = np.concatenate(metas, 0)
    metas = torch.FloatTensor(metas)
    meta_indexs = torch.LongTensor(meta_indexs)
    
    vecs = np.concatenate(vecs, 0)
    vecs = torch.FloatTensor(vecs)
    vec_indexs = torch.LongTensor(vec_indexs)
    
    frames = np.concatenate(frames, 0)
    frames = torch.FloatTensor(frames)
    frame_indexs = torch.LongTensor(frame_indexs)   

    audios = np.concatenate(audios, 0)
    audios = torch.FloatTensor(audios)
    audio_indexs = torch.LongTensor(audio_indexs)
    
    labels = torch.LongTensor(labels)

    return metas,meta_indexs,frames,frame_indexs, vecs,vec_indexs, audios,audio_indexs ,labels 
 

if __name__ == "__main__":
    from tqdm import tqdm
    import multiprocessing
    from math import ceil
    train_dataset, test_dataset = get_dataset('autodl-tmp/mini8k', 'autodl-tmp')

    def process(dataset, begin, end):
        for t in range(begin, end):
            print(f"processing {begin}/{t}/{end}")
            dataset[t]
        print(f"processing {begin}/{end} end!")

    chunksize = ceil(len(train_dataset) / 10)
    pool = multiprocessing.Pool(processes=10)
    results = [pool.apply_async(process, (train_dataset, i * chunksize, min((i + 1) * chunksize, len(train_dataset)))) for i in range(10)]
    pool.close()

    test_chunksize = ceil(len(test_dataset) / 10)
    test_pool = multiprocessing.Pool(processes=10)
    results = [test_pool.apply_async(process, (test_dataset,
                                          i * test_chunksize,
                                          min((i + 1) * test_chunksize, len(test_dataset)))) for i in range(10)]
    test_pool.close()
    pool.join()
    test_pool.join()






