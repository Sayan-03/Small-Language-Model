# Small Language Model

## Introduction

In this project, our primary goal is to develop a sophisticated yet efficient language model capable of auto-completing user-provided prompts. For this task we have used Transformer-Model architecture that uses multi-head attention module and a feed-forward neural network.

## Installing prerequisites

Before running the python files, make sure you have installed the required dependencies:

```bash
#Replace cu117 with the appropriate cuda version of your system
pip install pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

```
## The data extraction code
```bash
import os
import re
import random

class Extractor:
    def __init__(self,arr):
        self.train_root=arr[0]
        self.valid_root=arr[1]
        self.train_outputFile=arr[2]
        self.valid_outputFile=arr[3]
        self.vocabularyFile=arr[4]
        self.vocab=set()
    
    def _files_in_directory(self,root):
        files=[]
        for f_name in os.listdir(root):
            if f_name.endswith(".txt") and os.path.isfile(os.path.join(root,f_name)):
                files.append(f_name)
        return files
    
    def _strip_special_characters(self,word):
        pattern=r'[^a-zA-Z0-9\s]+'
        stripped_word=re.sub(pattern,'',word)
        return stripped_word
    
    def _get_words(self,text):
        words=text.split()
        list_of_words=[]
        for word in words:
            word=self._strip_special_characters(word)
            word=word.strip("~`!@#$%^&*()1234567890_-+={[]}\|'';:""/?.><,")
            if word:
                list_of_words.append(word)
        return list_of_words
    
    def get_vocabularyFile(self):
        print(f"Generating the vocabulary file..\n")
        train_files=self._files_in_directory(self.train_root)
        valid_files=self._files_in_directory(self.valid_root)
        total_train_files=len(train_files)
        total_valid_files=len(valid_files)
        
        with open(self.train_outputFile,"w",encoding="utf-8") as out_file:
            for f_name in train_files:
                file_path=os.path.join(train_root,f_name)
                with open(file_path,"r",encoding="utf-8") as in_file:
                    text=in_file.read()
                    out_file.write(text)
                    words=set(self._get_words(text))
                    self.vocab.update(words)

        with open(self.valid_outputFile,"w",encoding="utf-8") as out_file:
            for f_name in valid_files:
                file_path=os.path.join(valid_root,f_name)
                with open(file_path,"r",encoding="utf-8") as in_file:
                    text=in_file.read()
                    out_file.write(text)
                    words=set(self._get_words(text))
                    self.vocab.update(words)

        with open(self.vocabularyFile,"w",encoding="utf-8") as v_file:
            for word in self.vocab:
                v_file.write(word+'\n')
        print(f"Vocabulary File successfully generated!\n")
                
    def _seek_vocabularyFile(self,slice_num):
        with open(self.vocabularyFile,"r",encoding="utf-8") as f:
            text=f.read()
            words=sorted(self._get_words(text))
        vocab_size=len(words)
        print(f"Vocabulary File size: {vocab_size}\n")
        idx=random.randint(slice_num,vocab_size)
        print(f"Your Vocabulary File slice:\n {words[idx-slice_num:idx]}")

train_root="D:/LLM_Dataset/Books/Train"
valid_root="D:/LLM_Dataset/Books/Valid"
train_outputFile="D:/LLM_Dataset/output_train.txt"
valid_outputFile="D:/LLM_Dataset/output_valid.txt"
vocabularyFile="D:/LLM_Dataset/vocab.txt"

database=Extractor([train_root,valid_root,train_outputFile,valid_outputFile,vocabularyFile])

database.get_vocabularyFile()
```

## The main model code
```bash

import torch
import mmap
import random
import pickle
import re
import os

class smaLLLanguageModel:
    def __init__(self,vocabulary_file,train_opFile,valid_opFile):
        if torch.cuda.is_available():
            self.device='cuda'
            
        self.vocab_file=vocabulary_file
        self.train_outputFile=train_opFile
        self.valid_outputFile=valid_opFile
        
        self.block_size=8
        self.batch_size=32

        self.iters=100
        self.learning_rate=3e-4
        self.eval_iters=100
        self.dropout=0.2

        self.n_embd=50
        self.n_layer=1
        self.n_head=1
        
        self.vocab=[]
        self.word_to_int={}
        self.int_to_word={}
        
    def _strip_special_characters(self,word):
        pattern=r'[^a-zA-Z0-9\s]+'
        stripped_word=re.sub(pattern,'',word)
        return stripped_word
    
    def _get_words(self,text):
        words=text.split()
        list_of_words=[]
        for word in words:
            word=self._strip_special_characters(word)
            word=word.strip("~`!@#$%^&*()1234567890_-+={[]}\|'';:""/?.><,")
            if word:
                list_of_words.append(word)
        return list_of_words
    
    def _update_the_vocab(self,text):
        new_words=self._get_words(text)
        for word in new_words:
            if word not in self.vocab:
                self.vocab.append(word)
        self.vocab=sorted(self.vocab)
        new_word_indices={word:i for i,word in enumerate(self.vocab)}
        self.word_to_int.update(new_word_indices)
        self.int_to_word={i:word for word,i in self.word_to_int.items()}
   
    def _encoder(self,text):
        self._update_the_vocab(text)
        encoded_text=[]
        for word in self._get_words(text):
            encoded_text.append(self.word_to_int[word])
        return encoded_text
    
    def _decoder(self,arr):
        decoded_text=""
        for int_val in arr:
            word=self.int_to_word[int_val]
            decoded_text=decoded_text+word+" "
        return decoded_text
        
    def _get_vocabulary(self):
        with open(self.vocab_file,"r",encoding="utf-8") as f:
            text=f.read()
            self.vocab=sorted(set(self._get_words(text)))
        vocab_size=len(self.vocab)
        return vocab_size
        
    def _get_random_chunk(self,split):
        f_name=self.train_outputFile if split=="train" else self.valid_outputFile
        with open(f_name,"rb") as f:
            with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
                file_size=len(mm)
                start_pos=random.randint(0,(file_size)-self.block_size*self.batch_size)
                mm.seek(start_pos)
                block=mm.read(self.block_size*self.batch_size-1)
                decoded_block=block.decode("utf-8",errors="ignore").replace('\r','')
                data=torch.tensor(self._encoder(decoded_block),dtype=torch.long)
        return data
    
    def _get_batch(self,split):
        data=self._get_random_chunk(split)
        idx=torch.randint(len(data)-self.block_size,(self.batch_size,))
        x=torch.stack([data[i:i+self.block_size] for i in idx])
        y=torch.stack([data[i+1:i+1+self.block_size] for i in idx])
        x,y=x.to(self.device),y.to(self.device)
        return x,y
    
    @torch.no_grad()
    def _estimate_loss(self):
        out={}
        self.model.eval()
        for split in ["train","val"]:
            losses=torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X,Y=self._get_batch(split)
                _,loss=self.model(X,Y)
                losses[k]=loss.item()
            out[split]=losses.mean()
        self.model.train()
        return out
    
    def train(self):
        vocab_size=self._get_vocabulary()
        print(f"Vocabulary File accessed!\n")
        
        model=SLM(vocab_size)
        self.model=model.to(self.device)
        
        
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.learning_rate)
        
        print(f"Model training starts...\n")
        for iter_ in range(self.iters):
            xb,yb=self._get_batch("train")
            logits,loss=self.model.forward(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (iter_+1)%50==0:
                losses=self._estimate_loss()
                print(f'Epoch: [{iter_+1}/{self.iters}], Train loss: {losses["train"]:.5f}, Val loss: {losses["val"]:.5f}')
        print(f"\nModel trained!")
                
    def save_model(self,dir_,name):
        path=os.path.join(dir_,name)
        with open(path,"wb") as f:
            pickle.dump(self.model,f)
        print("Model Saved!")

class Head(torch.nn.Module):
    def __init__(self,head_size):
        super(Head,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.key=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,head_size,bias=False)
        self.query=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,head_size,bias=False)
        self.value=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(self.obj_smaLLLanguageModel.block_size,self.obj_smaLLLanguageModel.block_size)))
        self.dropout=torch.nn.Dropout(self.obj_smaLLLanguageModel.dropout)
        
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        wei=q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=torch.nn.functional.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        v=self.value(x)
        out=wei@v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,head_size):
        super(MultiHeadAttention,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.heads=torch.nn.ModuleList([Head(head_size) for _ in range(self.obj_smaLLLanguageModel.n_head)])
        self.proj=torch.nn.Linear(head_size*self.obj_smaLLLanguageModel.n_head,self.obj_smaLLLanguageModel.n_embd)
        self.dropout=torch.nn.Dropout(self.obj_smaLLLanguageModel.dropout)
        
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedForward(torch.nn.Module):
    def __init__(self):
        super(FeedForward,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.net=torch.nn.Sequential(
            torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,4*self.obj_smaLLLanguageModel.n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4*self.obj_smaLLLanguageModel.n_embd,self.obj_smaLLLanguageModel.n_embd),
            torch.nn.Dropout(self.obj_smaLLLanguageModel.dropout),
        )
        
    def forward(self,x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self):
        super(Block,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        head_size=self.obj_smaLLLanguageModel.n_embd//self.obj_smaLLLanguageModel.n_head
        self.aa=MultiHeadAttention(head_size)
        self.ffwd=FeedForward()
        self.ln1=torch.nn.LayerNorm(self.obj_smaLLLanguageModel.n_embd)
        self.ln2=torch.nn.LayerNorm(self.obj_smaLLLanguageModel.n_embd)
        
    def forward(self,x):
        y=self.aa(x)
        x=self.ln1(x+y)
        y=self.ffwd(x)
        x=self.ln2(x+y)
        return x

class SLM(torch.nn.Module):
    def __init__(self,vocab_size):
        super(SLM,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.embedding_Table=torch.nn.Embedding(vocab_size,self.obj_smaLLLanguageModel.n_embd)
        self.pos_embedding_Table=torch.nn.Embedding(self.obj_smaLLLanguageModel.block_size,self.obj_smaLLLanguageModel.n_embd)
        
        self.blocks=torch.nn.Sequential(*[Block() for _ in range(self.obj_smaLLLanguageModel.n_layer)])
        self.ln_f=torch.nn.LayerNorm(self.obj_smaLLLanguageModel.n_embd)
        self.lm_head=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        if isinstance(module,torch.nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,torch.nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
        
    def forward(self,index,targets=None):
        B,T=index.shape
            
        token_embeddings=self.embedding_Table(index)
        pos_embeddings=self.pos_embedding_Table(torch.arange(T,device=self.obj_smaLLLanguageModel.device))
        x=token_embeddings+pos_embeddings
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=torch.nn.functional.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,index,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss=self.forward(index)
            logits=logits[:,-1,:]
            probs=torch.nn.functional.softmax(logits,dim=-1)
            index_next=torch.multinomial(probs,num_samples=1)
            index=torch.cat((index,index_next),dim=1)
        return index

model=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")

model.train()

model.save_model("D:/LLM_Dataset/Model","Model_01.pkl")


```
## The bot code
```bash
import torch
import mmap
import random
import pickle
import re
import os

class smaLLLanguageModel:
    def __init__(self,vocabulary_file,train_opFile,valid_opFile):
        if torch.cuda.is_available():
            self.device='cuda'
            
        self.vocab_file=vocabulary_file
        self.train_outputFile=train_opFile
        self.valid_outputFile=valid_opFile
        
        self.block_size=8
        self.batch_size=32

        self.iters=100
        self.learning_rate=3e-4
        self.eval_iters=100
        self.dropout=0.2

        self.n_embd=50
        self.n_layer=1
        self.n_head=1
        
        self.vocab=[]
        self.word_to_int={}
        self.int_to_word={}
        
    def _strip_special_characters(self,word):
        pattern=r'[^a-zA-Z0-9\s]+'
        stripped_word=re.sub(pattern,'',word)
        return stripped_word
    
    def _get_words(self,text):
        words=text.split()
        list_of_words=[]
        for word in words:
            word=self._strip_special_characters(word)
            word=word.strip("~`!@#$%^&*()1234567890_-+={[]}\|'';:""/?.><,")
            if word:
                list_of_words.append(word)
        return list_of_words
    
    def _update_the_vocab(self,text):
        new_words=self._get_words(text)
        for word in new_words:
            if word not in self.vocab:
                self.vocab.append(word)
        self.vocab=sorted(self.vocab)
        new_word_indices={word:i for i,word in enumerate(self.vocab)}
        self.word_to_int.update(new_word_indices)
        self.int_to_word={i:word for word,i in self.word_to_int.items()}
            
    def _encoder(self,text):
        self._update_the_vocab(text)
        encoded_text=[]
        for word in self._get_words(text):
            encoded_text.append(self.word_to_int[word])
        return encoded_text
    
    def _decoder(self,arr):
        decoded_text=""
        for int_val in arr:
            word=self.int_to_word[int_val]
            decoded_text=decoded_text+word+" "
        return decoded_text
        
    def _get_vocabulary(self):
        with open(self.vocab_file,"r",encoding="utf-8") as f:
            text=f.read()
            self.vocab=sorted(set(self._get_words(text)))
        vocab_size=len(self.vocab)
        return vocab_size
    
    def load(self):
        vocab_size=self._get_vocabulary()
        model=SLM(vocab_size)
        print("Loading Model...\n")
        with open("D:/LLM_Dataset/Model/Model_01.pkl","rb") as f:
            model=pickle.load(f)
        print("Model successfully loaded!")
        self.model=model.to(self.device)
        
    def talk(self):
        rep=0
        while True:
            prompt=input("Prompt: ")
            context=torch.tensor(self._encoder(prompt),dtype=torch.long,device=self.device)
            generated_chars=self._decoder(self.model.generate(context.unsqueeze(0),max_new_tokens=5)[0].tolist())
            print(f'Completion: {generated_chars}\n')
            if rep==4:
                repeat=int(input("More? 1:YES  2:NO"))
                print(f"\n")
                if repeat==2:
                    break
    
class Head(torch.nn.Module):
    def __init__(self,head_size):
        super(Head,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.key=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,head_size,bias=False)
        self.query=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,head_size,bias=False)
        self.value=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(self.obj_smaLLLanguageModel.block_size,self.obj_smaLLLanguageModel.block_size)))
        self.dropout=torch.nn.Dropout(self.obj_smaLLLanguageModel.dropout)
        
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        wei=q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=torch.nn.functional.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        v=self.value(x)
        out=wei@v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,head_size):
        super(MultiHeadAttention,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.heads=torch.nn.ModuleList([Head(head_size) for _ in range(self.obj_smaLLLanguageModel.n_head)])
        self.proj=torch.nn.Linear(head_size*self.obj_smaLLLanguageModel.n_head,self.obj_smaLLLanguageModel.n_embd)
        self.dropout=torch.nn.Dropout(self.obj_smaLLLanguageModel.dropout)
        
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedForward(torch.nn.Module):
    def __init__(self):
        super(FeedForward,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.net=torch.nn.Sequential(
            torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,4*self.obj_smaLLLanguageModel.n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4*self.obj_smaLLLanguageModel.n_embd,self.obj_smaLLLanguageModel.n_embd),
            torch.nn.Dropout(self.obj_smaLLLanguageModel.dropout),
        )
        
    def forward(self,x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self):
        super(Block,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        head_size=self.obj_smaLLLanguageModel.n_embd//self.obj_smaLLLanguageModel.n_head
        self.aa=MultiHeadAttention(head_size)
        self.ffwd=FeedForward()
        self.ln1=torch.nn.LayerNorm(self.obj_smaLLLanguageModel.n_embd)
        self.ln2=torch.nn.LayerNorm(self.obj_smaLLLanguageModel.n_embd)
        
    def forward(self,x):
        y=self.aa(x)
        x=self.ln1(x+y)
        y=self.ffwd(x)
        x=self.ln2(x+y)
        return x

class SLM(torch.nn.Module):
    def __init__(self,vocab_size):
        super(SLM,self).__init__()
        self.obj_smaLLLanguageModel=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")
        self.embedding_Table=torch.nn.Embedding(vocab_size,self.obj_smaLLLanguageModel.n_embd)
        self.pos_embedding_Table=torch.nn.Embedding(self.obj_smaLLLanguageModel.block_size,self.obj_smaLLLanguageModel.n_embd)
        
        self.blocks=torch.nn.Sequential(*[Block() for _ in range(self.obj_smaLLLanguageModel.n_layer)])
        self.ln_f=torch.nn.LayerNorm(self.obj_smaLLLanguageModel.n_embd)
        self.lm_head=torch.nn.Linear(self.obj_smaLLLanguageModel.n_embd,vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        if isinstance(module,torch.nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,torch.nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
        
    def forward(self,index,targets=None):
        B,T=index.shape
            
        token_embeddings=self.embedding_Table(index)
        pos_embeddings=self.pos_embedding_Table(torch.arange(T,device=self.obj_smaLLLanguageModel.device))
        x=token_embeddings+pos_embeddings
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=torch.nn.functional.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,index,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss=self.forward(index)
            logits=logits[:,-1,:]
            probs=torch.nn.functional.softmax(logits,dim=-1)
            index_next=torch.multinomial(probs,num_samples=1)
            index=torch.cat((index,index_next),dim=1)
        return index

chatbot=smaLLLanguageModel("D:/LLM_Dataset/vocab.txt","D:/LLM_Dataset/output_train.txt","D:/LLM_Dataset/output_valid.txt")

chatbot.load()

chatbot.talk()


```


## Clone the repository
```bash
git clone https://github.com/Sayan-03/Small-Language-Model.git
cd Small-Language-Model
```
