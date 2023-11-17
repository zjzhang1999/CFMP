#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from pretrainedmodels.models.fbresnet import *

import torch
import torch.nn as nn
import numpy as np
class FullModel(torch.nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        
        self.cross_feature = 512
        self.Fq = torch.nn.Sequential(torch.nn.Linear(600, self.cross_feature, bias=False),
                                      torch.nn.ReLU())
        self.Fk = torch.nn.Sequential(torch.nn.Linear(600, self.cross_feature, bias=False),
                                      torch.nn.ReLU())
        self.Fv = torch.nn.Sequential(torch.nn.Linear(600, self.cross_feature, bias=False),
                                      torch.nn.ReLU())

        self.Vq = torch.nn.Sequential(torch.nn.Linear(768, self.cross_feature, bias=False),
                                      torch.nn.ReLU())
        self.Vk = torch.nn.Sequential(torch.nn.Linear(768, self.cross_feature, bias=False),
                                      torch.nn.ReLU())
        self.Vv = torch.nn.Sequential(torch.nn.Linear(768, self.cross_feature, bias=False),
                                      torch.nn.ReLU())

        self.Aq = torch.nn.Sequential(torch.nn.Linear(128, self.cross_feature, bias=False),
                                      torch.nn.ReLU())
        self.Ak = torch.nn.Sequential(torch.nn.Linear(128, self.cross_feature, bias=False),
                                      torch.nn.ReLU())
        self.Av = torch.nn.Sequential(torch.nn.Linear(128, self.cross_feature, bias=False),
                                      torch.nn.ReLU())

        self.ff_f = torch.nn.ModuleList([torch.nn.Linear(self.cross_feature, self.cross_feature, bias=False) for i in range(2)])
        self.ff_v = torch.nn.ModuleList([torch.nn.Linear(self.cross_feature, self.cross_feature, bias=False) for i in range(2)])
        self.ff_a = torch.nn.ModuleList([torch.nn.Linear(self.cross_feature, self.cross_feature, bias=False) for i in range(2)])
 
        
        self.out = torch.nn.Sequential(torch.nn.Linear(self.cross_feature  + 2  , 768, bias=False),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.3),
                                       torch.nn.Linear(768, 512, bias=False),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.3),
                                       torch.nn.Linear(512, 64,bias=False),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(64, 1))

        self.dropout = nn.Dropout(p=0.3) 

        


    def forward(self, metas,meta_indexs,frames, frame_indexs, vecs, vec_indexs,audios,audio_indexs):
        bn = torch.unique(frame_indexs).cpu().numpy()
        bfs = torch.zeros([bn.shape[0]], device=frames.device)

        for n in bn:
            
            m_features = metas[meta_indexs == n]
            
            
            meta_goal = m_features[0][1].unsqueeze(0).unsqueeze(1)/1500  
            meta_start = m_features[0][2].unsqueeze(0).unsqueeze(1)
            meta_end = m_features[0][3].unsqueeze(0).unsqueeze(1)
            dura = meta_end - meta_start
            #print(meta_goal)
            if dura>8000:
                dura = torch.tensor(100).to('cuda:0').unsqueeze(0).unsqueeze(1)
            dura = dura / 100
            meta_create = m_features[0][4].unsqueeze(0).unsqueeze(1)/5
         
            


            b_features = frames[frame_indexs == n] #（1，600）   
            f_q = torch.mean(self.Fq(b_features), dim=0).view(-1, 1)  #（512，n）
            f_k = self.Fk(b_features)  #(n,512)
            f_v = self.Fv(b_features).permute(1, 0) #(cross,n)
            attn = torch.nn.functional.softmax(torch.mm(f_k, f_q) / self.cross_feature) #(n,1)
            f_q_p = torch.mm(f_v, attn) #(cross,1)


            v_features = vecs[vec_indexs == n]
            v_q = torch.mean(self.Vq(v_features), dim=0).view(-1, 1)
            v_k = self.Vk(v_features)
            v_v = self.Vv(v_features).permute(1, 0)
            attn = torch.nn.functional.softmax(torch.mm(v_k, v_q) / self.cross_feature)
            v_q_p = torch.mm(v_v, attn)



            a_features = audios[audio_indexs == n]
            a_features = a_features.view(1, 128)
            a_q = torch.mean(self.Aq(a_features), dim=0).view(-1, 1)
            a_k = self.Ak(a_features)
            a_v = self.Av(a_features).permute(1, 0)
            attn = torch.nn.functional.softmax(torch.mm(a_k, a_q) / self.cross_feature)
            a_q_p = torch.mm(a_v, attn)



            for i in range(2):
                # for video
                attn = torch.nn.functional.softmax(torch.mm(f_k, v_q_p) / self.cross_feature, dim=0)
                f_c = torch.mm(f_v, attn)
                f_q_n = self.ff_f[i](f_c.permute(1, 0)).permute(1, 0) + f_q_p

                # for text
                attn = torch.nn.functional.softmax(torch.mm(v_k, v_q_p) / self.cross_feature, dim=0)
                v_c = torch.mm(v_v, attn)
                v_q_n = self.ff_v[i](v_c.permute(1, 0)).permute(1, 0) + v_q_p

                # for audio
                
                attn = torch.nn.functional.softmax(torch.mm(a_k, v_q_p) / self.cross_feature, dim=0)
                a_c = torch.mm(a_v, attn)
                a_q_n = self.ff_a[i](a_c.permute(1, 0)).permute(1, 0) + a_q_p
                
                # switch
                
                f_q_p = self.dropout(f_q_n)
                v_q_p = self.dropout(v_q_n)
                a_q_p = self.dropout(a_q_n)
                
                f_q_p = f_q_n
                v_q_p = v_q_n
                a_q_p = a_q_n
            
            mean = torch.mean(a_q_p)
            std = torch.std(a_q_p)
            normalized_a_q_p = (a_q_p - mean) / std
            
           
            
            features = torch.cat([f_q_p.permute(1, 0), v_q_p.permute(1, 0),normalized_a_q_p.permute(1, 0),meta_goal,meta_create,dura], dim=1)
            #features = torch.cat([v_q_p.permute(1, 0),meta_goal,meta_create], dim=1)
            features = self.out(features)
            bfs[n] = features.sigmoid()
        return bfs



