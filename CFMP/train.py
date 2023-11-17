#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 4level -train
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn
from torch.utils.data.dataloader import DataLoader
from xxdataset import *
from tqdm import tqdm
from xfull_model import FullModel
from torchvggish import vggish


import matplotlib.pyplot as plt
import ffmpeg
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings('ignore', message="Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.")
warnings.filterwarnings('ignore', message="UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.")



MAX_EPOCH = 20


if __name__ == '__main__':
    model = FullModel().cuda()
    train_dataset, val_dataset = get_dataset('./autodl-tmp/Kick30k/train_val', 'autodl-tmp')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=collate,
                                  drop_last=True,)

    val_dataloader = DataLoader(val_dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=collate,
                                 drop_last=True,)
    #loss_func = torch.nn.BCELoss()
    loss_func = torch.nn.CrossEntropyLoss()
    optm = torch.optim.Adam(model.parameters(), lr=0.0001)

    acclst = []
    met1,met2,met3,met4,met5,met6 = [],[],[],[],[],[]
    

    for epoch in range(MAX_EPOCH):
        
        model.train()
        total_accuracy = 0 
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_batches = 0 
        
        for metas,meta_indexs, frames, frame_indexs, vecs, vec_indexs, audios,audio_indexs,labels,pseudos in tqdm(train_dataloader):
            if len(frames.shape) == 1:
                continue
            #print(pseudos)
            labels = pseudos.cuda()
            #print('labels:',labels)
            
            feature = model(metas.cuda(),meta_indexs.cuda(),frames.cuda(), frame_indexs.cuda(), vecs.cuda(), vec_indexs.cuda(), audios.cuda(), audio_indexs.cuda())
            #print('feature:',feature)

            loss = loss_func(feature, labels.cuda().long())
            optm.zero_grad()
            loss.backward()
            optm.step()

            feature_np = feature.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            predicted_labels = feature_np.argmax(axis=1)

   
            accuracy = accuracy_score(labels_np, predicted_labels)
            precision = precision_score(labels_np, predicted_labels, average='macro')
            recall = recall_score(labels_np, predicted_labels, average='macro')
            f1 = f1_score(labels_np, predicted_labels, average='macro')

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_batches += 1

        average_accuracy = total_accuracy / total_batches
        average_precision = total_precision / total_batches
        average_recall = total_recall / total_batches
        average_f1 = total_f1 / total_batches

        print("Epoch:", epoch)
        print("Average Accuracy:", average_accuracy)
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1)
             
        
        model.eval()
        total_accuracy = 0 
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_batches = 0 
        
        for metas,meta_indexs, frames, frame_indexs, vecs, vec_indexs, audios,audio_indexs,labels,pseudos in tqdm(val_dataloader):
            if len(frames.shape) == 1:
                continue
            #print(pseudos)
            labels = pseudos.cuda()
            #print('labels:',labels)
            
            feature = model(metas.cuda(),meta_indexs.cuda(),frames.cuda(), frame_indexs.cuda(), vecs.cuda(), vec_indexs.cuda(), audios.cuda(), audio_indexs.cuda())
            #print('feature:',feature)


            feature_np = feature.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            predicted_labels = feature_np.argmax(axis=1)

   
            accuracy = accuracy_score(labels_np, predicted_labels)
            precision = precision_score(labels_np, predicted_labels, average='macro')
            recall = recall_score(labels_np, predicted_labels, average='macro')
            f1 = f1_score(labels_np, predicted_labels, average='macro')

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_batches += 1

        average_accuracy = total_accuracy / total_batches
        average_precision = total_precision / total_batches
        average_recall = total_recall / total_batches
        average_f1 = total_f1 / total_batches

        print("Epoch:", epoch)
        print("Val Accuracy:", average_accuracy)
        print("Val Precision:", average_precision)
        print("Val Recall:", average_recall)
        print("Val F1 Score:", average_f1)

# 2level


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn
from torch.utils.data.dataloader import DataLoader
from xxdataset import *
from tqdm import tqdm
from xfull_model import FullModel
from torchvggish import vggish


import matplotlib.pyplot as plt
import ffmpeg
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings('ignore', message="Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.")


MAX_EPOCH = 30


if __name__ == '__main__':
    model = FullModel().cuda()
    train_dataset, val_dataset = get_dataset('./autodl-tmp/Kick30k/train_val', 'autodl-tmp')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=collate,
                                  drop_last=True,)

    val_dataloader = DataLoader(val_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=collate,
                                 drop_last=True,)
    loss_func = torch.nn.BCELoss()
    optm = torch.optim.Adam(model.parameters(), lr=0.0001)

    acclst = []
    met1,met2,met3,met4,met5,met6 = [],[],[],[],[],[]
    

    for epoch in range(MAX_EPOCH):
        
        model.train()
        true_labels = []
        predicted_probs = []

        train_results = []
        train_losses = []  
        
        for metas,meta_indexs, frames, frame_indexs, vecs, vec_indexs, audios,audio_indexs,labels in tqdm(train_dataloader):
            if len(frames.shape) == 1:
                continue

            labels = labels.cuda()

            feature = model(metas.cuda(),meta_indexs.cuda(),frames.cuda(), frame_indexs.cuda(), vecs.cuda(), vec_indexs.cuda(), audios.cuda(), audio_indexs.cuda())
            loss = loss_func(feature, labels.cuda().float())
            optm.zero_grad()
            loss.backward()
            optm.step()
                       
            predicted_labels = (feature > 0.5).long()          
            true_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(predicted_labels.cpu().numpy())     
            correct = (predicted_labels == labels)
            
            train_results.append(correct)
            train_losses.append(loss.item())   
            
        train_loss = sum(train_losses) / len(train_losses)
        train_results = torch.cat(train_results)
        train_acc = torch.sum(train_results).item() / train_results.shape[0]
        
        print(f"Epoch {epoch + 1}:")
        print('train_loss:',round(train_loss,4)*100,'train_acc:',round(train_acc,4)*100)
        met1.append(train_loss)
        met2.append(train_acc)

        model.eval()
        test_results = []
        true_labels = []
        predicted_probs = []

        for metas,meta_indexs,frames, frame_indexs, vecs, vec_indexs, audios,audio_indexs,labels in tqdm(val_dataloader):
            if len(frames.shape) == 1:
                continue
            labels = labels.cuda()

            feature = model(metas.cuda(),meta_indexs.cuda(),frames.cuda(), frame_indexs.cuda(), vecs.cuda(), vec_indexs.cuda(), audios.cuda(), audio_indexs.cuda())
            predicted_labels = (feature > 0.5).long()
            true_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(predicted_labels.cpu().numpy())
            correct = (predicted_labels == labels)
            test_results.append(correct)

        test_results = torch.cat(test_results)
        test_acc = torch.sum(test_results).item() / test_results.shape[0]

        true_labels = np.array(true_labels)
        predicted_probs = np.array(predicted_probs)

        acc2 = accuracy_score(true_labels, predicted_probs)
        precision = precision_score(true_labels, predicted_probs, average='binary')
        recall = recall_score(true_labels, predicted_probs, average='binary')
        f1 = f1_score(true_labels, predicted_probs, average='binary')

    
        print("Test_Accuracy2:", round(acc2,4)* 100)
        print("Test_Precision:", round(precision,4)*100)
        print("Test_Recall:", round(recall,4)*100)
        print("Test_F1-Score:", round(f1,4)*100)
        
        met3.append(acc2)
        met4.append(precision)
        met5.append(recall)
        met6.append(f1)





