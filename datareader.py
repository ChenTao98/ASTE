import json,os,argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import defaultdict
import nltk
import sys

# unused cur_max 45
aspect_left=["[CLS]","[unused1]","[unused2]","[unused3]"]
aspect_mid=["[unused37]","[unused38]","[unused39]"]
aspect_right=["[unused4]","[unused5]","[unused6]","[SEP]"]

opinion_left=["[CLS]","[unused7]","[unused8]","[unused9]"]
opinion_right=["[unused10]","[unused11]","[unused12]","[SEP]"]

aspect_sentiment_left=["[CLS]","[unused25]","[unused26]","[unused27]"]
aspect_sentiment_mid=["[unused40]","[unused41]","[unused42]"]
aspect_sentiment_right=["[unused28]","[unused29]","[unused30]","[SEP]"]



opinion_sentiment_left=["[CLS]","[unused31]","[unused32]","[unused33]"]
opinion_sentiment_mid=["[unused43]","[unused44]","[unused45]"]
opinion_sentiment_right=["[unused34]","[unused35]","[unused36]","[SEP]"]

polarity_map={"pos":1,"neg":2,"neu":0,1:1,0:0,2:2,"none":0}
polarity_id_map_text={0:"neutral",1:"positive",2:"negative"}



with open("pos_tag.json") as in_fp:
    pos_tag_map=json.load(in_fp)

class Example:
    def __init__(self,sentence,aspect_start,aspect_end,aspect,opinion,opinion_start,opinion_end,polarity,opinion_map_aspect_start,opinion_map_aspect_end,dual_opinion_start,dual_opinion_end):
        self.sentence=sentence
        self.aspect_start=aspect_start
        self.aspect_end=aspect_end
        self.polarity=polarity
        self.aspect=aspect
        self.opinion_start=opinion_start
        self.opinion_end=opinion_end
        
        self.opinion=opinion
        self.opinion_map_aspect_start=opinion_map_aspect_start
        self.opinion_map_aspect_end=opinion_map_aspect_end
        self.dual_opinion_start=dual_opinion_start
        self.dual_opinion_end=dual_opinion_end
    
    def read_example(self):
        return (self.sentence,self.aspect_start,self.aspect_end,self.aspect,self.opinion,self.opinion_start,self.opinion_end,self.polarity,self.opinion_map_aspect_start,self.opinion_map_aspect_end,self.dual_opinion_start,self.dual_opinion_end)

class EvalExample:
    def __init__(self,sentence,aspect_list,opinion_list,pairs,line):
        self.sentence=sentence
        self.aspect_list=aspect_list
        self.opinion_list=opinion_list
        self.pairs=pairs
        self.line=line
    
    def read_example(self,return_line=False):
        if(return_line):
            return (self.sentence,self.aspect_list,self.opinion_list,self.pairs,self.line)
        return (self.sentence,self.aspect_list,self.opinion_list,self.pairs)

def get_aspect_map_opinion_query(aspect,tokenizer,sentiment,args):
    if(args.train_binary):
        query=aspect_left+tokenizer.tokenize(aspect)+aspect_right
    else:
        query=aspect_left+tokenizer.tokenize(aspect)+aspect_mid+tokenizer.tokenize(polarity_id_map_text[sentiment])+aspect_right
    # query=aspect_left+["[unused4]","[unused5]","[unused6]"]+tokenizer.tokenize(aspect)+["[SEP]"]
    # query=["CLS"]+tokenizer.tokenize("please find the opinion item of "+aspect)+["[SEP]"]
    return query


def get_aspect_map_sentiment_query(aspect,tokenizer):
    # query=aspect_sentiment_left+tokenizer.tokenize(aspect)+aspect_sentiment_right
    # query=aspect_sentiment_left+["sentiment","of"]+tokenizer.tokenize(aspect)+["is","[MASK]"]
    query=aspect_sentiment_left+tokenizer.tokenize(aspect)+aspect_sentiment_mid+["[MASK]"]
    mask_index=len(query)-1
    query=query+aspect_sentiment_right
    # query=aspect_left+["[unused4]","[unused5]","[unused6]"]+tokenizer.tokenize(aspect)+["[SEP]"]
    # query=["CLS"]+tokenizer.tokenize("please find the opinion item of "+aspect)+["[SEP]"]
    return query,mask_index

def get_opinion_map_aspect_query(opinion,tokenizer):
    query=opinion_left+tokenizer.tokenize(opinion)+opinion_right
    # query=aspect_left+["[unused4]","[unused5]","[unused6]"]+tokenizer.tokenize(aspect)+["[SEP]"]
    # query=["CLS"]+tokenizer.tokenize("please find the opinion item of "+aspect)+["[SEP]"]
    return query

def get_opinion_map_sentiment_query(opinion,tokenizer):
    # query=opinion_sentiment_left+tokenizer.tokenize(opinion)+opinion_sentiment_right
    # query=opinion_sentiment_left+["sentiment","of"]+tokenizer.tokenize(opinion)+["is","[MASK]"]
    query=opinion_sentiment_left+tokenizer.tokenize(opinion)+opinion_sentiment_mid+["[MASK]"]
    mask_index=len(query)-1
    query=query+opinion_sentiment_right
    # query=aspect_left+["[unused4]","[unused5]","[unused6]"]+tokenizer.tokenize(aspect)+["[SEP]"]
    # query=["CLS"]+tokenizer.tokenize("please find the opinion item of "+aspect)+["[SEP]"]
    return query,mask_index

def get_aspect_query(tokenizer):
    # return ["CLS"]+tokenizer.tokenize("find the aspect terms in the text")+["[SEP]"]
    return ["[CLS]","[unused19]","[unused20]","[unused21]","aspect","[unused22]","[unused23]","[unused24]","[SEP]"]
    

def get_dual_opinion_query(tokenizer):
    return ["[CLS]","[unused13]","[unused14]","[unused15]","opinion","[unused16]","[unused17]","[unused18]","[SEP]"]

def convert_evaluate_aspect_example_to_feature(sentence,tokenizer,max_length):
    input_ids=np.zeros((1,max_length),dtype=np.int64)
    token_type_ids=np.zeros((1,max_length),dtype=np.int64)
    attention_mask=np.zeros((1,max_length),dtype=np.int64)
    
    pos_tag_id=np.zeros((1,max_length),dtype=np.int64)
    sentence_list=sentence.split(" ")
    nltk_pos=nltk.pos_tag(sentence_list)
    nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
    
    
    all_token=get_aspect_query(tokenizer)
    
    cur_pos_tag=[1 for x in all_token]
    
    
    first_sentence_lenth=len(all_token)
    for jj in range(len(sentence_list)):
        cur_token_pos_tag_id=nltk_pos[jj]
        tokenizer_result=tokenizer.tokenize(sentence_list[jj])
        all_token=all_token+tokenizer_result
        for _ in tokenizer_result:
            cur_pos_tag.append(cur_token_pos_tag_id)
    cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 
    all_token=all_token[:max_length-1]+["[SEP]"]      
    sentence_list=tokenizer.convert_tokens_to_ids(all_token)
    input_ids[0,:len(sentence_list)]=sentence_list
    attention_mask[0,:len(sentence_list)]=1
    token_type_ids[0,first_sentence_lenth:len(sentence_list)]=1
    pos_tag_id[0,:len(sentence_list)]=cur_pos_tag
    sentence_length=len(sentence_list)-first_sentence_lenth-1
    return torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),first_sentence_lenth,sentence_length,torch.from_numpy(pos_tag_id)

def convert_evaluate_dual_opinion_example_to_feature(sentence,tokenizer,max_length):
    input_ids=np.zeros((1,max_length),dtype=np.int64)
    token_type_ids=np.zeros((1,max_length),dtype=np.int64)
    attention_mask=np.zeros((1,max_length),dtype=np.int64)
    
    pos_tag_id=np.zeros((1,max_length),dtype=np.int64)
    sentence_list=sentence.split(" ")
    nltk_pos=nltk.pos_tag(sentence_list)
    nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
    
    
    all_token=get_dual_opinion_query(tokenizer)
    
    cur_pos_tag=[1 for x in all_token]
    
    
    first_sentence_lenth=len(all_token)
    for jj in range(len(sentence_list)):
        cur_token_pos_tag_id=nltk_pos[jj]
        tokenizer_result=tokenizer.tokenize(sentence_list[jj])
        all_token=all_token+tokenizer_result
        for _ in tokenizer_result:
            cur_pos_tag.append(cur_token_pos_tag_id)
    cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 
    all_token=all_token[:max_length-1]+["[SEP]"]      
    sentence_list=tokenizer.convert_tokens_to_ids(all_token)
    input_ids[0,:len(sentence_list)]=sentence_list
    attention_mask[0,:len(sentence_list)]=1
    token_type_ids[0,first_sentence_lenth:len(sentence_list)]=1
    pos_tag_id[0,:len(sentence_list)]=cur_pos_tag
    sentence_length=len(sentence_list)-first_sentence_lenth-1
    return torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),first_sentence_lenth,sentence_length,torch.from_numpy(pos_tag_id)


def convert_evaluate_opinion_example_to_feature(sentence,aspect,tokenizer,max_length,sentiment,args):
    input_ids=np.zeros((1,max_length),dtype=np.int64)
    token_type_ids=np.zeros((1,max_length),dtype=np.int64)
    attention_mask=np.zeros((1,max_length),dtype=np.int64)
    pos_tag_id=np.zeros((1,max_length),dtype=np.int64)
    sentence_list=sentence.split(" ")
    nltk_pos=nltk.pos_tag(sentence_list)
    nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
    all_token=get_aspect_map_opinion_query(aspect,tokenizer,sentiment,args)
    cur_pos_tag=[1 for x in all_token]
    first_sentence_lenth=len(all_token)
    for jj in range(len(sentence_list)):
        cur_token_pos_tag_id=nltk_pos[jj]
        tokenizer_result=tokenizer.tokenize(sentence_list[jj])
        all_token=all_token+tokenizer_result
        for _ in tokenizer_result:
            cur_pos_tag.append(cur_token_pos_tag_id)
    cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 

    all_token=all_token[:max_length-1]+["[SEP]"]      
    sentence_list=tokenizer.convert_tokens_to_ids(all_token)
    input_ids[0,:len(sentence_list)]=sentence_list
    attention_mask[0,:len(sentence_list)]=1
    pos_tag_id[0,:len(sentence_list)]=cur_pos_tag
    token_type_ids[0,first_sentence_lenth:len(sentence_list)]=1
    sentence_length=len(sentence_list)-first_sentence_lenth-1
    return torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),first_sentence_lenth,sentence_length,torch.from_numpy(pos_tag_id)


def convert_evaluate_aspect_sentiment_example_to_feature(sentence,aspect,tokenizer,max_length):
    input_ids=np.zeros((1,max_length),dtype=np.int64)
    token_type_ids=np.zeros((1,max_length),dtype=np.int64)
    attention_mask=np.zeros((1,max_length),dtype=np.int64)
    pos_tag_id=np.zeros((1,max_length),dtype=np.int64)
    sentence_list=sentence.split(" ")
    nltk_pos=nltk.pos_tag(sentence_list)
    nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
    all_token,mask_index=get_aspect_map_sentiment_query(aspect,tokenizer)
    cur_pos_tag=[1 for x in all_token]
    first_sentence_lenth=len(all_token)
    for jj in range(len(sentence_list)):
        cur_token_pos_tag_id=nltk_pos[jj]
        tokenizer_result=tokenizer.tokenize(sentence_list[jj])
        all_token=all_token+tokenizer_result
        for _ in tokenizer_result:
            cur_pos_tag.append(cur_token_pos_tag_id)
    cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 

    all_token=all_token[:max_length-1]+["[SEP]"]      
    sentence_list=tokenizer.convert_tokens_to_ids(all_token)
    input_ids[0,:len(sentence_list)]=sentence_list
    attention_mask[0,:len(sentence_list)]=1
    pos_tag_id[0,:len(sentence_list)]=cur_pos_tag
    token_type_ids[0,first_sentence_lenth:len(sentence_list)]=1
    sentence_length=len(sentence_list)-first_sentence_lenth-1
    return torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),first_sentence_lenth,sentence_length,torch.from_numpy(pos_tag_id),mask_index

def convert_evaluate_opinion_sentiment_example_to_feature(sentence,opinion,tokenizer,max_length):
    input_ids=np.zeros((1,max_length),dtype=np.int64)
    token_type_ids=np.zeros((1,max_length),dtype=np.int64)
    attention_mask=np.zeros((1,max_length),dtype=np.int64)
    pos_tag_id=np.zeros((1,max_length),dtype=np.int64)
    sentence_list=sentence.split(" ")
    nltk_pos=nltk.pos_tag(sentence_list)
    nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
    all_token,mask_index=get_opinion_map_sentiment_query(opinion,tokenizer)

    cur_pos_tag=[1 for x in all_token]
    
    first_sentence_lenth=len(all_token)
    for jj in range(len(sentence_list)):
        cur_token_pos_tag_id=nltk_pos[jj]
        tokenizer_result=tokenizer.tokenize(sentence_list[jj])
        all_token=all_token+tokenizer_result
        for _ in tokenizer_result:
            cur_pos_tag.append(cur_token_pos_tag_id)
    cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 

    all_token=all_token[:max_length-1]+["[SEP]"]      
    sentence_list=tokenizer.convert_tokens_to_ids(all_token)
    input_ids[0,:len(sentence_list)]=sentence_list
    attention_mask[0,:len(sentence_list)]=1
    pos_tag_id[0,:len(sentence_list)]=cur_pos_tag
    token_type_ids[0,first_sentence_lenth:len(sentence_list)]=1
    sentence_length=len(sentence_list)-first_sentence_lenth-1
    return torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),first_sentence_lenth,sentence_length,torch.from_numpy(pos_tag_id),mask_index


def convert_evaluate_aspect_by_opinion_example_to_feature(sentence,opinion,tokenizer,max_length):
    input_ids=np.zeros((1,max_length),dtype=np.int64)
    token_type_ids=np.zeros((1,max_length),dtype=np.int64)
    attention_mask=np.zeros((1,max_length),dtype=np.int64)
    pos_tag_id=np.zeros((1,max_length),dtype=np.int64)
    sentence_list=sentence.split(" ")
    nltk_pos=nltk.pos_tag(sentence_list)
    nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
    all_token=get_opinion_map_aspect_query(opinion,tokenizer)

    cur_pos_tag=[1 for x in all_token]
    
    first_sentence_lenth=len(all_token)
    for jj in range(len(sentence_list)):
        cur_token_pos_tag_id=nltk_pos[jj]
        tokenizer_result=tokenizer.tokenize(sentence_list[jj])
        all_token=all_token+tokenizer_result
        for _ in tokenizer_result:
            cur_pos_tag.append(cur_token_pos_tag_id)
    cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 

    all_token=all_token[:max_length-1]+["[SEP]"]      
    sentence_list=tokenizer.convert_tokens_to_ids(all_token)
    input_ids[0,:len(sentence_list)]=sentence_list
    attention_mask[0,:len(sentence_list)]=1
    pos_tag_id[0,:len(sentence_list)]=cur_pos_tag
    token_type_ids[0,first_sentence_lenth:len(sentence_list)]=1
    sentence_length=len(sentence_list)-first_sentence_lenth-1
    return torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),first_sentence_lenth,sentence_length,torch.from_numpy(pos_tag_id)

def read_evaluate_example(file_path,tokenizer):
    all_examples=list()
    with open(file_path) as in_fp:
        for line in tqdm(in_fp,desc="read example"):
            sentence,labels=line.strip().lower().split("####")
            labels=labels.replace("(","[").replace(")","]").replace("'","\"")
            labels=json.loads(labels)
            pairs=list()
            aspect_list=list()
            opinion_list=list()
            split_sentence=sentence.split(" ")
            # print(sentence)
            for cur_label in labels:
                # print(cur_label)
                polarity=polarity_map[cur_label[2]]
                aspect_token_list=list()
                for token in split_sentence[cur_label[0][0]:cur_label[0][-1]+1]:
                    aspect_token_list=aspect_token_list+tokenizer.tokenize(token)
                aspect=tokenizer.convert_tokens_to_string(aspect_token_list)
                opinion_token_list=list()
                for token in split_sentence[cur_label[1][0]:cur_label[1][-1]+1]:
                    opinion_token_list=opinion_token_list+tokenizer.tokenize(token)
                opinion=tokenizer.convert_tokens_to_string(opinion_token_list)

                # aspect=" ".join(split_sentence[cur_label[0][0]:cur_label[0][-1]+1])
                # opinion=" ".join(split_sentence[cur_label[1][0]:cur_label[1][-1]+1])
                aspect_list.append(aspect)
                opinion_list.append(opinion)
                pairs.append((aspect,opinion,int(polarity)))
            all_examples.append(EvalExample(sentence,aspect_list,opinion_list,pairs,line.strip()))
    return all_examples

def read_from_file(file_path):
    all_examples=list()
    with open(file_path) as in_fp:
        for line in in_fp:
            # print(line)
            sentence,labels=line.strip().lower().split("####")
            labels=labels.replace("(","[").replace(")","]").replace("'","\"")
            labels=json.loads(labels)
            aspect_map_opinion=defaultdict(list)
            aspect_start_span_index=set()
            # print(sentence)
            split_sentence=sentence.split(" ")
            for cur_label in labels:
                # print(cur_label)
                aspect_start_span_index.add((cur_label[0][0],cur_label[0][-1]))
                start=cur_label[1][0]
                end=cur_label[1][-1]
                polarity=polarity_map[cur_label[2]]
                aspect=" ".join(split_sentence[cur_label[0][0]:cur_label[0][-1]+1])
                aspect_map_opinion[aspect].append((start,end,int(polarity)))

            opinion_map_aspect=defaultdict(list)
            opinion_start_span_index=set()
            for cur_label in labels:
                # print(cur_label)
                opinion_start_span_index.add((cur_label[1][0],cur_label[1][-1]))
                start=cur_label[0][0]
                end=cur_label[0][-1]
                polarity=polarity_map[cur_label[2]]
                opinion=" ".join(split_sentence[cur_label[1][0]:cur_label[1][-1]+1])
                aspect=" ".join(split_sentence[start:end+1])
                opinion_map_aspect[opinion].append((start,end,int(polarity),aspect))
            
            aspect_start_span_index=sorted(list(aspect_start_span_index),key=lambda x:x[0])
            aspect_start=list()
            aspect_end=list()
            for start,end in aspect_start_span_index:
                aspect_start.append(start)
                aspect_end.append(end)
            opinion_start_span_index=sorted(list(opinion_start_span_index),key=lambda x:x[0])
            dual_opinion_start=list()
            dual_opinion_end=list()
            for start,end in opinion_start_span_index:
                dual_opinion_start.append(start)
                dual_opinion_end.append(end)
            
            
            
            
            for opinion_k,opinion_v in sorted(opinion_map_aspect.items()):
                opinion_map_aspect_start=list()
                opinion_map_aspect_end=list()
                opinion_polarity_list=list()
                for temp_start,temp_end,temp_polarity,temp_aspect in opinion_v:
                    opinion_map_aspect_start.append(temp_start)
                    opinion_map_aspect_end.append(temp_end)
                    opinion_polarity_list.append(temp_polarity)
                for temp_start,temp_end,temp_polarity,temp_aspect in opinion_v:
                    start_list=list()
                    end_list=list()
                    polarity_list=list()
                    for start,end,polarity in aspect_map_opinion[temp_aspect]:
                        start_list.append(start)
                        end_list.append(end)
                        polarity_list.append(polarity)
                    all_examples.append(Example(sentence,aspect_start,aspect_end,temp_aspect,opinion_k,start_list,end_list,polarity_list,opinion_map_aspect_start,opinion_map_aspect_end,dual_opinion_start,dual_opinion_end))
                
    
            # for k,v in aspect_map_opinion.items():
            #     start_list=list()
            #     end_list=list()
            #     polarity_list=list()
            #     for start,end,polarity in v:
            #         start_list.append(start)
            #         end_list.append(end)
            #         polarity_list.append(polarity)
            #     all_examples.append(Example(sentence,aspect_start,aspect_end,k,None,start_list,end_list,polarity_list))
    return all_examples

def convert_example_to_features(tokenizer,examples,max_length,args):
    print(len(examples))
    input_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    token_type_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    attention_mask=np.zeros((len(examples),max_length),dtype=np.int64)
    start_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    end_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    global_label=np.zeros((len(examples),1,max_length,max_length),dtype=np.int64)
    first_index=np.zeros(len(examples),dtype=np.int64)
    sentence_length=np.zeros(len(examples),dtype=np.int64)
    pos_tag_id=np.zeros((len(examples),max_length),dtype=np.int64)
    

    aspect_sentiment_input_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_sentiment_token_type_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_sentiment_attention_mask=np.zeros((len(examples),max_length),dtype=np.int64)  
    aspect_sentiment_pos_tag_id=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_sentiment_mask_label=np.full((len(examples),max_length),-100,dtype=np.int64)
    aspect_sentiment_mask_index=np.zeros((len(examples)),dtype=np.int64)

    opinion_sentiment_input_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_sentiment_token_type_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_sentiment_attention_mask=np.zeros((len(examples),max_length),dtype=np.int64) 
    opinion_sentiment_pos_tag_id=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_sentiment_mask_label=np.full((len(examples),max_length),-100,dtype=np.int64)
    opinion_sentiment_mask_index=np.zeros((len(examples)),dtype=np.int64)
    

    aspect_input_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_token_type_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_attention_mask=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_start_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_end_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    aspect_global_label=np.zeros((len(examples),1,max_length,max_length),dtype=np.int64)
    aspect_first_index=np.zeros(len(examples),dtype=np.int64)
    aspect_sentence_length=np.zeros(len(examples),dtype=np.int64)
    aspect_pos_tag_id=np.zeros((len(examples),max_length),dtype=np.int64)
    

    dual_opinion_input_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    dual_opinion_token_type_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    dual_opinion_attention_mask=np.zeros((len(examples),max_length),dtype=np.int64)
    dual_opinion_start_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    dual_opinion_end_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    dual_opinion_global_label=np.zeros((len(examples),1,max_length,max_length),dtype=np.int64)
    dual_opinion_first_index=np.zeros(len(examples),dtype=np.int64)
    dual_opinion_sentence_length=np.zeros(len(examples),dtype=np.int64)
    dual_opinion_pos_tag_id=np.zeros((len(examples),max_length),dtype=np.int64)

    opinion_map_aspect_input_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_map_aspect_token_type_ids=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_map_aspect_attention_mask=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_map_aspect_start_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_map_aspect_end_pos=np.zeros((len(examples),max_length),dtype=np.int64)
    opinion_map_aspect_global_label=np.zeros((len(examples),1,max_length,max_length),dtype=np.int64)
    opinion_map_aspect_first_index=np.zeros(len(examples),dtype=np.int64)
    opinion_map_aspect_sentence_length=np.zeros(len(examples),dtype=np.int64)
    opinion_map_aspect_pos_tag_id=np.zeros((len(examples),max_length),dtype=np.int64)
    
    polarity_list=np.zeros((len(examples)),dtype=np.int64)

    test_max_len=0
    aspect_text_list=list()
    opinion_text_list=list()
    for ii,cur_example in tqdm(enumerate(examples),desc="convert examples to features"):
        sentence,aspect_start,aspect_end,aspect,opinion,opinion_start,opinion_end,polarity,opinion_map_aspect_start,opinion_map_aspect_end,dual_opinion_start,dual_opinion_end=cur_example.read_example()
        
        polarity_list[ii]=polarity[0]
        
        aspect_text_list.append(aspect)
        sentence_list=sentence.split(" ")
        
        nltk_pos=nltk.pos_tag(sentence_list)
        nltk_pos=[pos_tag_map.get(pos[1],2) for pos in nltk_pos]
        
        all_token=get_aspect_map_opinion_query(aspect,tokenizer,polarity[0],args)

        first_sentence_lenth=len(all_token)
        first_index[ii]=first_sentence_lenth
        cur_start_set=set(opinion_start)
        cur_end_set=set(opinion_end)
        start_index_map=dict()
        end_index_map=dict()
        cur_pos_tag=[1 for x in all_token]
        for jj in range(len(sentence_list)):
            cur_token_pos_tag_id=nltk_pos[jj]
            if(jj in cur_start_set):
                start_index_map[jj]=len(all_token)
                start_pos[ii,len(all_token)]=1
            tokenizer_result=tokenizer.tokenize(sentence_list[jj])
            all_token=all_token+tokenizer_result
            for _ in tokenizer_result:
                cur_pos_tag.append(cur_token_pos_tag_id)
            if(jj in cur_end_set):
                end_index_map[jj]=len(all_token)
                end_pos[ii,len(all_token)]=1
        for cur_start,cur_end in zip(opinion_start,opinion_end):
            global_label[ii,0,start_index_map[cur_start],end_index_map[cur_end]]=1
        all_token=all_token[:max_length-1]+["[SEP]"] 
        cur_pos_tag=cur_pos_tag[:max_length-1]+[1]     
        sentence_list=tokenizer.convert_tokens_to_ids(all_token)
        test_max_len=max(len(sentence_list),test_max_len)
        input_ids[ii,:len(sentence_list)]=sentence_list
        attention_mask[ii,:len(sentence_list)]=1
        token_type_ids[ii,first_sentence_lenth:len(sentence_list)]=1
        sentence_length[ii]=len(sentence_list)-first_sentence_lenth-1
        pos_tag_id[ii,:len(sentence_list)]=cur_pos_tag
        
        # 转aspect的mrc
        sentence_list=sentence.split(" ")
        all_token=get_aspect_query(tokenizer)
        first_sentence_lenth=len(all_token)
        aspect_first_index[ii]=first_sentence_lenth
        cur_start_set=set(aspect_start)
        cur_end_set=set(aspect_end)
        start_index_map=dict()
        end_index_map=dict()
        cur_pos_tag=[1 for x in all_token]
        for jj in range(len(sentence_list)):
            cur_token_pos_tag_id=nltk_pos[jj]
            if(jj in cur_start_set):
                start_index_map[jj]=len(all_token)
                aspect_start_pos[ii,len(all_token)]=1
            tokenizer_result=tokenizer.tokenize(sentence_list[jj])
            all_token=all_token+tokenizer_result
            for _ in tokenizer_result:
                cur_pos_tag.append(cur_token_pos_tag_id)
            if(jj in cur_end_set):
                end_index_map[jj]=len(all_token)
                aspect_end_pos[ii,len(all_token)]=1
        for cur_start,cur_end in zip(aspect_start,aspect_end):
            aspect_global_label[ii,0,start_index_map[cur_start],end_index_map[cur_end]]=1
        all_token=all_token[:max_length-1]+["[SEP]"]      
        cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 
        sentence_list=tokenizer.convert_tokens_to_ids(all_token)
        test_max_len=max(len(sentence_list),test_max_len)
        aspect_input_ids[ii,:len(sentence_list)]=sentence_list
        aspect_attention_mask[ii,:len(sentence_list)]=1
        aspect_token_type_ids[ii,first_sentence_lenth:len(sentence_list)]=1
        aspect_sentence_length[ii]=len(sentence_list)-first_sentence_lenth-1
        aspect_pos_tag_id[ii,:len(sentence_list)]=cur_pos_tag
        
        # 转dual_opinion的mrc
        sentence_list=sentence.split(" ")
        all_token=get_dual_opinion_query(tokenizer)
        first_sentence_lenth=len(all_token)
        dual_opinion_first_index[ii]=first_sentence_lenth
        cur_start_set=set(dual_opinion_start)
        cur_end_set=set(dual_opinion_end)
        start_index_map=dict()
        end_index_map=dict()
        cur_pos_tag=[1 for x in all_token]
        for jj in range(len(sentence_list)):
            cur_token_pos_tag_id=nltk_pos[jj]
            if(jj in cur_start_set):
                start_index_map[jj]=len(all_token)
                dual_opinion_start_pos[ii,len(all_token)]=1
            tokenizer_result=tokenizer.tokenize(sentence_list[jj])
            all_token=all_token+tokenizer_result
            for _ in tokenizer_result:
                cur_pos_tag.append(cur_token_pos_tag_id)
            if(jj in cur_end_set):
                end_index_map[jj]=len(all_token)
                dual_opinion_end_pos[ii,len(all_token)]=1
        for cur_start,cur_end in zip(dual_opinion_start,dual_opinion_end):
            dual_opinion_global_label[ii,0,start_index_map[cur_start],end_index_map[cur_end]]=1
        all_token=all_token[:max_length-1]+["[SEP]"]      
        cur_pos_tag=cur_pos_tag[:max_length-1]+[1] 
        sentence_list=tokenizer.convert_tokens_to_ids(all_token)
        test_max_len=max(len(sentence_list),test_max_len)
        dual_opinion_input_ids[ii,:len(sentence_list)]=sentence_list
        dual_opinion_attention_mask[ii,:len(sentence_list)]=1
        dual_opinion_token_type_ids[ii,first_sentence_lenth:len(sentence_list)]=1
        dual_opinion_sentence_length[ii]=len(sentence_list)-first_sentence_lenth-1
        dual_opinion_pos_tag_id[ii,:len(sentence_list)]=cur_pos_tag
        
        
        # 转opinion 抽aspect的mrc
        sentence_list=sentence.split(" ")
        all_token=get_opinion_map_aspect_query(opinion,tokenizer)
        first_sentence_lenth=len(all_token)
        opinion_map_aspect_first_index[ii]=first_sentence_lenth
        cur_start_set=set(opinion_map_aspect_start)
        cur_end_set=set(opinion_map_aspect_end)
        start_index_map=dict()
        end_index_map=dict()
        cur_pos_tag=[1 for x in all_token]
        for jj in range(len(sentence_list)):
            cur_token_pos_tag_id=nltk_pos[jj]
            if(jj in cur_start_set):
                start_index_map[jj]=len(all_token)
                opinion_map_aspect_start_pos[ii,len(all_token)]=1
            tokenizer_result=tokenizer.tokenize(sentence_list[jj])
            all_token=all_token+tokenizer_result
            for _ in tokenizer_result:
                cur_pos_tag.append(cur_token_pos_tag_id)
            if(jj in cur_end_set):
                end_index_map[jj]=len(all_token)
                opinion_map_aspect_end_pos[ii,len(all_token)]=1
        for cur_start,cur_end in zip(opinion_map_aspect_start,opinion_map_aspect_end):
            opinion_map_aspect_global_label[ii,0,start_index_map[cur_start],end_index_map[cur_end]]=1
        all_token=all_token[:max_length-1]+["[SEP]"]  
        cur_pos_tag=cur_pos_tag[:max_length-1]+[1]     
        sentence_list=tokenizer.convert_tokens_to_ids(all_token)
        test_max_len=max(len(sentence_list),test_max_len)
        opinion_map_aspect_input_ids[ii,:len(sentence_list)]=sentence_list
        opinion_map_aspect_attention_mask[ii,:len(sentence_list)]=1
        opinion_map_aspect_token_type_ids[ii,first_sentence_lenth:len(sentence_list)]=1
        opinion_map_aspect_sentence_length[ii]=len(sentence_list)-first_sentence_lenth-1
        opinion_map_aspect_pos_tag_id[ii,:len(sentence_list)]=cur_pos_tag
        
        sentence_list=sentence.split(" ")
        all_token,mask_index=get_aspect_map_sentiment_query(aspect,tokenizer)
        first_sentence_lenth=len(all_token)
        cur_pos_tag=[1 for x in all_token]
        for jj in range(len(sentence_list)):
            cur_token_pos_tag_id=nltk_pos[jj]
            tokenizer_result=tokenizer.tokenize(sentence_list[jj])
            all_token=all_token+tokenizer_result
            for _ in tokenizer_result:
                cur_pos_tag.append(cur_token_pos_tag_id)
        all_token=all_token[:max_length-1]+["[SEP]"]  
        cur_pos_tag=cur_pos_tag[:max_length-1]+[1]     
        sentence_list=tokenizer.convert_tokens_to_ids(all_token)
        aspect_sentiment_input_ids[ii,:len(sentence_list)]=sentence_list
        
        # print(sentence_list[mask_index])
        sentence_list[mask_index]=tokenizer.convert_tokens_to_ids(polarity_id_map_text[polarity[0]])
        aspect_sentiment_mask_label[ii,:len(sentence_list)]=sentence_list
        aspect_sentiment_mask_index[ii]=mask_index
        # print(sentence_list[mask_index])
        
        aspect_sentiment_attention_mask[ii,:len(sentence_list)]=1
        aspect_sentiment_token_type_ids[ii,first_sentence_lenth:len(sentence_list)]=1
        aspect_sentiment_pos_tag_id[ii,:len(sentence_list)]=cur_pos_tag

        sentence_list=sentence.split(" ")
        all_token,mask_index=get_opinion_map_sentiment_query(opinion,tokenizer)
        first_sentence_lenth=len(all_token)
        cur_pos_tag=[1 for x in all_token]
        for jj in range(len(sentence_list)):
            cur_token_pos_tag_id=nltk_pos[jj]
            tokenizer_result=tokenizer.tokenize(sentence_list[jj])
            all_token=all_token+tokenizer_result
            for _ in tokenizer_result:
                cur_pos_tag.append(cur_token_pos_tag_id)
        all_token=all_token[:max_length-1]+["[SEP]"]  
        cur_pos_tag=cur_pos_tag[:max_length-1]+[1]     
        sentence_list=tokenizer.convert_tokens_to_ids(all_token)
        opinion_sentiment_input_ids[ii,:len(sentence_list)]=sentence_list
        
        # print(sentence_list[mask_index])
        sentence_list[mask_index]=tokenizer.convert_tokens_to_ids(polarity_id_map_text[polarity[0]])
        opinion_sentiment_mask_label[ii,:len(sentence_list)]=sentence_list
        opinion_sentiment_mask_index[ii]=mask_index
        # print(sentence_list[mask_index])
        
        opinion_sentiment_attention_mask[ii,:len(sentence_list)]=1
        opinion_sentiment_token_type_ids[ii,first_sentence_lenth:len(sentence_list)]=1
        opinion_sentiment_pos_tag_id[ii,:len(sentence_list)]=cur_pos_tag
        
        
    print(test_max_len)
    return (torch.from_numpy(input_ids),torch.from_numpy(token_type_ids),torch.from_numpy(attention_mask),torch.from_numpy(global_label),torch.from_numpy(aspect_input_ids),torch.from_numpy(aspect_token_type_ids),torch.from_numpy(aspect_attention_mask),torch.from_numpy(aspect_global_label),torch.from_numpy(dual_opinion_input_ids),torch.from_numpy(dual_opinion_token_type_ids),torch.from_numpy(dual_opinion_attention_mask),torch.from_numpy(dual_opinion_global_label),torch.from_numpy(opinion_map_aspect_input_ids),torch.from_numpy(opinion_map_aspect_token_type_ids),torch.from_numpy(opinion_map_aspect_attention_mask),torch.from_numpy(opinion_map_aspect_global_label),torch.from_numpy(polarity_list),torch.from_numpy(pos_tag_id),torch.from_numpy(aspect_pos_tag_id),torch.from_numpy(dual_opinion_pos_tag_id),torch.from_numpy(opinion_map_aspect_pos_tag_id),torch.from_numpy(aspect_sentiment_input_ids),torch.from_numpy(aspect_sentiment_token_type_ids),torch.from_numpy(aspect_sentiment_attention_mask),torch.from_numpy(aspect_sentiment_pos_tag_id),torch.from_numpy(opinion_sentiment_input_ids),torch.from_numpy(opinion_sentiment_token_type_ids),torch.from_numpy(opinion_sentiment_attention_mask),torch.from_numpy(opinion_sentiment_pos_tag_id),torch.from_numpy(aspect_sentiment_mask_label),torch.from_numpy(opinion_sentiment_mask_label),torch.from_numpy(aspect_sentiment_mask_index),torch.from_numpy(opinion_sentiment_mask_index))

class MrcOpinionDataset(Dataset):
    def __init__(self,file_path,tokenizer,args) -> None:
        examples=read_from_file(file_path)
        features=convert_example_to_features(tokenizer,examples,args.max_length)
        self.input_ids=features[0]
        self.token_type_ids=features[1]
        self.attention_mask=features[2]
        self.start_pos=features[3]
        self.end_pos=features[4]
        self.first_index=features[5]
        self.sentence_length=features[6]

        self.aspect_input_ids=features[7]
        self.aspect_token_type_ids=features[8]
        self.aspect_attention_mask=features[9]
        self.aspect_start_pos=features[10]
        self.aspect_end_pos=features[11]
        self.aspect_first_index=features[12]
        self.aspect_sentence_length=features[13]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.token_type_ids[index],self.attention_mask[index],self.start_pos[index],self.end_pos[index],self.first_index[index],self.sentence_length[index]

class MrcAspectOpinionDataset(Dataset):
    def __init__(self,file_path,tokenizer,args) -> None:
        examples=read_from_file(file_path)
        features=convert_example_to_features(tokenizer,examples,args.max_length,args)
        self.input_ids=features[0]
        self.token_type_ids=features[1]
        self.attention_mask=features[2]
        self.global_pos=features[3]

        self.aspect_input_ids=features[4]
        self.aspect_token_type_ids=features[5]
        self.aspect_attention_mask=features[6]
        self.aspect_global_pos=features[7]
        
        self.dual_opinion_input_ids=features[8]
        self.dual_opinion_token_type_ids=features[9]
        self.dual_opinion_attention_mask=features[10]
        self.dual_opinion_global_pos=features[11]
        
        self.opinion_map_aspect_input_ids=features[12]
        self.opinion_map_aspect_token_type_ids=features[13]
        self.opinion_map_aspect_attention_mask=features[14]
        self.opinion_map_aspect_global_pos=features[15]
        
        self.polarity_list=features[16]
        self.pos_tag_id=features[17]
        self.aspect_pos_tag_id=features[18]
        self.dual_opinion_pos_tag_id=features[19]
        self.opinion_map_aspect_pos_tag_id=features[20]
        
        self.aspect_sentiment_input_ids=features[21]
        self.aspect_sentiment_token_type_ids=features[22]
        self.aspect_sentiment_attention_mask=features[23]
        self.aspect_sentiment_pos_tag_id=features[24]
        
        self.opinion_sentiment_input_ids=features[25]
        self.opinion_sentiment_token_type_ids=features[26]
        self.opinion_sentiment_attention_mask=features[27]
        self.opinion_sentiment_pos_tag_id=features[28]
        
        self.aspect_sentiment_mask_label=features[29]
        self.opinion_sentiment_mask_label=features[30]
        
        self.aspect_sentiment_mask_index=features[31]
        self.opinion_sentiment_mask_index=features[32]
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.token_type_ids[index],self.attention_mask[index],self.global_pos[index],self.aspect_input_ids[index],self.aspect_token_type_ids[index],self.aspect_attention_mask[index],self.aspect_global_pos[index],self.dual_opinion_input_ids[index],self.dual_opinion_token_type_ids[index],self.dual_opinion_attention_mask[index],self.dual_opinion_global_pos[index],self.opinion_map_aspect_input_ids[index],self.opinion_map_aspect_token_type_ids[index],self.opinion_map_aspect_attention_mask[index],self.opinion_map_aspect_global_pos[index],self.polarity_list[index],self.pos_tag_id[index],self.aspect_pos_tag_id[index],self.dual_opinion_pos_tag_id[index],self.opinion_map_aspect_pos_tag_id[index],self.aspect_sentiment_input_ids[index],self.aspect_sentiment_token_type_ids[index],self.aspect_sentiment_attention_mask[index],self.aspect_sentiment_pos_tag_id[index],self.opinion_sentiment_input_ids[index],self.opinion_sentiment_token_type_ids[index],self.opinion_sentiment_attention_mask[index],self.opinion_sentiment_pos_tag_id[index],self.aspect_sentiment_mask_label[index],self.opinion_sentiment_mask_label[index],self.aspect_sentiment_mask_index[index],self.opinion_sentiment_mask_index[index]



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--max_length",type=int,default=128)
    args=parser.parse_args()
    file_path="/home/gdmlab/chentao/aspect_opinion/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
    result=read_from_file(file_path)
    print(len(result))
    print(result[0].read_example())
    print(result[3].read_example())
    model_path="/home/gdmlab/chentao/model/bert-base-uncased"

    tokenizer=BertTokenizer.from_pretrained(model_path)
    dataset=MrcOpinionDataset(file_path,tokenizer,args)
    print(len(dataset))
    print(dataset.__getitem__(0))