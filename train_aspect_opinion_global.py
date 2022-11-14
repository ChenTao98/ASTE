
import json
from tqdm import tqdm
from datareader import MrcAspectOpinionDataset,read_evaluate_example,convert_evaluate_aspect_example_to_feature,convert_evaluate_opinion_example_to_feature,convert_evaluate_aspect_by_opinion_example_to_feature,convert_evaluate_dual_opinion_example_to_feature,convert_evaluate_opinion_sentiment_example_to_feature,convert_evaluate_aspect_sentiment_example_to_feature,polarity_id_map_text
from model import MrcAspectOpinionModel,GlobalPointerModel
import utils
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import os,sys,argparse,random
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import collections
from shutil import copyfile
logging.basicConfig(level=logging.INFO)

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


polarity_id_map_text_token_id=[-1,-1,-1]

def map_polarity_id_to_token_id(tokenizer):
    for k,v in polarity_id_map_text.items():
        polarity_id_map_text_token_id[k]=tokenizer.convert_tokens_to_ids(v)
        
class MrcOpinionTrainer:
    def __init__(self,tokenizer,args,total_step):
        self.model=GlobalPointerModel(args,tokenizer,sentiment_label_index=polarity_id_map_text_token_id)
        self.model.to(args.device)
        no_decay = ['bias', 'LayerNorm.weight']
        word_emb_weight = ["word_embeddings.weight"]
        # optimizer_grouped_parameters = [
        # {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in word_emb_weight)],
        #  'weight_decay': args.weight_decay,"lr":5e-5},
        # {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay) and not any(nd in n for nd in word_emb_weight))],
        #  'weight_decay': args.weight_decay},
        # {'params': [p for n, p in self.model.named_parameters() if (any(nd in n for nd in no_decay) and not any(nd in n for nd in word_emb_weight))], 'weight_decay': 0.0}]

        optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,"lr":1e-3}]
        
        self.bert_optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr)
        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_optimizer, num_warmup_steps=0.1*total_step,
                                                num_training_steps=total_step)
        self.best_f1=0
    
    def train_one_epoch(self,cur_epoch,train_dataloader,args):
        self.model.train()
        loop=tqdm(train_dataloader,desc="train {}".format(cur_epoch))
        total_loss=0
        count=0
        for input_ids,token_type_ids,attention_mask,global_pos,aspect_input_ids,aspect_token_type_ids,aspect_attention_mask,aspect_global_pos,dual_opinion_input_ids,dual_opinion_token_type_ids,dual_opinion_attention_mask,dual_opinion_global_pos,opinion_map_aspect_input_ids,opinion_map_aspect_token_type_ids,opinion_map_aspect_attention_mask,opinion_map_aspect_global_pos,polarity_label,pos_tag_id,aspect_pos_tag_id,dual_opinion_pos_tag_id,opinion_map_aspect_pos_tag_id,aspect_sentiment_input_ids,aspect_sentiment_token_type_ids,aspect_sentiment_attention_mask,aspect_sentiment_pos_tag_id,opinion_sentiment_input_ids,opinion_sentiment_token_type_ids,opinion_sentiment_attention_mask,opinion_sentiment_pos_tag_id,aspect_sentiment_mask_label,opinion_sentiment_mask_label,aspect_sentiment_mask_index,opinion_sentiment_mask_index in loop:
            if(args.use_pos_tag):
                pos_tag_id=pos_tag_id.to(args.device)
                aspect_pos_tag_id=aspect_pos_tag_id.to(args.device)
                dual_opinion_pos_tag_id=dual_opinion_pos_tag_id.to(args.device)
                opinion_map_aspect_pos_tag_id=opinion_map_aspect_pos_tag_id.to(args.device)
                aspect_sentiment_pos_tag_id=aspect_sentiment_pos_tag_id.to(args.device)
                opinion_sentiment_pos_tag_id=opinion_sentiment_pos_tag_id.to(args.device)
            else:
                pos_tag_id=None
                aspect_pos_tag_id=None
                opinion_map_aspect_pos_tag_id=None
                dual_opinion_pos_tag_id=None
                aspect_sentiment_pos_tag_id=None
                opinion_sentiment_pos_tag_id=None    
            
            if(args.train_binary):
                logits,aspect_map_opinion_loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),global_pos.to(args.device),mode="aspect_map_opinion",pos_tag_id=pos_tag_id)
                opinion_map_aspect_losgits,opinion_map_aspect_loss=self.model(opinion_map_aspect_input_ids.to(args.device),opinion_map_aspect_token_type_ids.to(args.device),opinion_map_aspect_attention_mask.to(args.device),opinion_map_aspect_global_pos.to(args.device),mode="opinion_map_aspect",pos_tag_id=opinion_map_aspect_pos_tag_id)
            else:
                logits,aspect_sentiment_loss=self.model(aspect_sentiment_input_ids.to(args.device),aspect_sentiment_token_type_ids.to(args.device),aspect_sentiment_attention_mask.to(args.device),None,polarity_label=polarity_label.to(args.device),mode="aspect_sentiment",pos_tag_id=aspect_sentiment_pos_tag_id,mask_index=aspect_sentiment_mask_index)
                logits,opinion_sentiment_loss=self.model(opinion_sentiment_input_ids.to(args.device),opinion_sentiment_token_type_ids.to(args.device),opinion_sentiment_attention_mask.to(args.device),None,polarity_label=polarity_label.to(args.device),mode="opinion_sentiment",pos_tag_id=opinion_sentiment_pos_tag_id,mask_index=opinion_sentiment_mask_index)
                logits,aspect_map_opinion_loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),global_pos.to(args.device),polarity_label=None,mode="aspect_map_opinion",pos_tag_id=pos_tag_id)
                opinion_map_aspect_losgits,opinion_map_aspect_loss=self.model(opinion_map_aspect_input_ids.to(args.device),opinion_map_aspect_token_type_ids.to(args.device),opinion_map_aspect_attention_mask.to(args.device),opinion_map_aspect_global_pos.to(args.device),polarity_label=None,mode="opinion_map_aspect",pos_tag_id=opinion_map_aspect_pos_tag_id)
            aspect_losgits,aspect_loss=self.model(aspect_input_ids.to(args.device),aspect_token_type_ids.to(args.device),aspect_attention_mask.to(args.device),aspect_global_pos.to(args.device),mode="aspect",pos_tag_id=aspect_pos_tag_id)
            dual_opinion_losgits,dual_opinion_loss=self.model(dual_opinion_input_ids.to(args.device),dual_opinion_token_type_ids.to(args.device),dual_opinion_attention_mask.to(args.device),dual_opinion_global_pos.to(args.device),mode="dual_opinion",pos_tag_id=dual_opinion_pos_tag_id)
            
            

            # loss=(opinion_loss*0.5+opinion_map_aspect_loss*0.5)*(1/3)+aspect_loss*(1/3)+polarity_loss*(1/3)
            # loss_1=aspect_map_opinion_loss+aspect_loss+dual_opinion_loss+opinion_map_aspect_loss+torch.pow((aspect_loss+aspect_map_opinion_loss-dual_opinion_loss-opinion_map_aspect_loss),2)
            
            # loss_1=opinion_loss+aspect_loss+dual_opinion_loss+opinion_map_aspect_loss
            # loss_1=torch.pow((aspect_loss+opinion_loss-dual_opinion_loss-opinion_map_aspect_loss),2)
            if(args.train_binary):
                dual_loss=torch.pow((aspect_loss+aspect_map_opinion_loss-dual_opinion_loss-opinion_map_aspect_loss),2)
                loss=aspect_loss+aspect_map_opinion_loss+dual_opinion_loss+opinion_map_aspect_loss+dual_loss
            else:
                dual_loss=torch.pow((aspect_loss+aspect_sentiment_loss+aspect_map_opinion_loss-dual_opinion_loss-opinion_map_aspect_loss-opinion_sentiment_loss),2)
                # loss=aspect_loss+aspect_sentiment_loss+aspect_map_opinion_loss+dual_opinion_loss+opinion_map_aspect_loss+opinion_sentiment_loss+dual_loss
                loss=aspect_loss+aspect_sentiment_loss+aspect_map_opinion_loss+dual_opinion_loss+opinion_map_aspect_loss+opinion_sentiment_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.bert_optimizer.step()
            self.bert_scheduler.step()
            self.bert_optimizer.zero_grad()
            total_loss+=loss.item()
            count+=1
            loop.set_postfix(loss=total_loss/count)
    
    def predict_aspect(self,sentence,args,tokenizer,return_index=False):
        input_ids,token_type_ids,attention_mask,first_sentence_lenth,sentence_length,pos_tag_id=convert_evaluate_aspect_example_to_feature(sentence,tokenizer,args.max_length)
        if(args.use_pos_tag):
            pos_tag_id=pos_tag_id.to(args.device)
        else:
            pos_tag_id=None
        logits,loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),mode="aspect",pos_tag_id=pos_tag_id)
        span_start_end,logits_list=utils.extract_one_by_global_logits(logits[0][0],sentence_length,first_sentence_lenth,args)
        aspect_list=utils.convert_span_start_end_to_text(span_start_end,input_ids[0].numpy().tolist(),tokenizer,first_sentence_lenth)
        if(return_index):
            return (span_start_end,aspect_list)
        return aspect_list,logits_list
    def predict_dual_opinion(self,sentence,args,tokenizer,return_index=False):
        input_ids,token_type_ids,attention_mask,first_sentence_lenth,sentence_length,pos_tag_id=convert_evaluate_dual_opinion_example_to_feature(sentence,tokenizer,args.max_length)
        if(args.use_pos_tag):
            pos_tag_id=pos_tag_id.to(args.device)
        else:
            pos_tag_id=None
        logits,loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),mode="dual_opinion",pos_tag_id=pos_tag_id)
        span_start_end,logits_list=utils.extract_one_by_global_logits(logits[0][0],sentence_length,first_sentence_lenth,args)
        dual_opinion_list=utils.convert_span_start_end_to_text(span_start_end,input_ids[0].numpy().tolist(),tokenizer,first_sentence_lenth)
        if(return_index):
            return (span_start_end,dual_opinion_list)
        return dual_opinion_list,logits_list
    def predict_opinion_by_aspect(self,sentence,aspect,args,tokenizer,return_index=False):
        
        input_ids,token_type_ids,attention_mask,first_sentence_lenth,sentence_length,pos_tag_id,mask_index=convert_evaluate_aspect_sentiment_example_to_feature(sentence,aspect,tokenizer,args.max_length)
        if(args.use_pos_tag):
            pos_tag_id=pos_tag_id.to(args.device)
        else:
            pos_tag_id=None
        polarity_logits,polarity_loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),polarity_label="predict",mode="aspect_sentiment",pos_tag_id=pos_tag_id,mask_index=mask_index)
        polarity_label=torch.argmax(polarity_logits,dim=-1)[0]
        # print(polarity_label.shape)
        
        input_ids,token_type_ids,attention_mask,first_sentence_lenth,sentence_length,pos_tag_id=convert_evaluate_opinion_example_to_feature(sentence,aspect,tokenizer,args.max_length,polarity_label.item(),args)
        if(args.use_pos_tag):
            pos_tag_id=pos_tag_id.to(args.device)
        else:
            pos_tag_id=None
        logits,loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),polarity_label=None,mode="aspect_map_opinion",pos_tag_id=pos_tag_id)
        span_start_end,logits_list=utils.extract_one_by_global_logits(logits[0][0],sentence_length,first_sentence_lenth,args)
        opinion_list=utils.convert_span_start_end_to_text(span_start_end,input_ids[0].numpy().tolist(),tokenizer,first_sentence_lenth)
        if(return_index):
            return (span_start_end,opinion_list,polarity_label)
        return opinion_list,polarity_label,logits_list

    def predict_aspect_by_opinion(self,sentence,opinion,args,tokenizer,return_index=False):

        input_ids,token_type_ids,attention_mask,first_sentence_lenth,sentence_length,pos_tag_id,mask_index=convert_evaluate_opinion_sentiment_example_to_feature(sentence,opinion,tokenizer,args.max_length)
        if(args.use_pos_tag):
            pos_tag_id=pos_tag_id.to(args.device)
        else:
            pos_tag_id=None
        polarity_logits,polarity_loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),polarity_label="predict",mode="opinion_sentiment",pos_tag_id=pos_tag_id,mask_index=mask_index)
        polarity_label=torch.argmax(polarity_logits,dim=-1)[0]


        input_ids,token_type_ids,attention_mask,first_sentence_lenth,sentence_length,pos_tag_id=convert_evaluate_aspect_by_opinion_example_to_feature(sentence,opinion,tokenizer,args.max_length)
        if(args.use_pos_tag):
            pos_tag_id=pos_tag_id.to(args.device)
        else:
            pos_tag_id=None
        logits,loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),polarity_label=None,mode="opinion_map_aspect",pos_tag_id=pos_tag_id)
        span_start_end,logits_list=utils.extract_one_by_global_logits(logits[0][0],sentence_length,first_sentence_lenth,args)
        opinion_list=utils.convert_span_start_end_to_text(span_start_end,input_ids[0].numpy().tolist(),tokenizer,first_sentence_lenth)
        if(return_index):
            return (span_start_end,polarity_label,opinion_list)
        return opinion_list,polarity_label,logits_list
    
    def evaluate(self,cur_epoch,eval_dataloader,args,tokenizer):
        self.model.eval()
        with torch.no_grad():
            pred_num=0
            gold_num=0
            correct_num=0
            aspect_pred_num=0
            aspect_gold_num=0
            aspect_correct_num=0
            
            dual_opinion_pred_num=0
            dual_opinion_gold_num=0
            dual_opinion_correct_num=0
            
            test_aspect_count=0
            opinion_predict_aspect_count=0
            out_result=list()
            for example in tqdm(eval_dataloader):
                sentence,aspect_list,dual_opinion_list,pairs=example.read_example()
                if(args.train_binary):
                    pairs=[(x[0],x[1]) for x in pairs]

                aspect_list=set(aspect_list)
                dual_opinion_list=set(dual_opinion_list)
                aspect_gold_num+=len(aspect_list)
                dual_opinion_gold_num+=len(dual_opinion_list)
                predict_aspect_list,aspect_logits_list=self.predict_aspect(sentence,args,tokenizer)
                predict_aspect_list=set(predict_aspect_list)
                aspect_pred_num+=len(predict_aspect_list)
                aspect_correct_num+=len(predict_aspect_list&aspect_list)
                predict_result=list()
                # for (aspect,aspect_logits) in zip(predict_aspect_list,aspect_logits_list):
                #     predict_opinion_list,polarity_label,opinion_logits_list=self.predict_opinion_by_aspect(sentence,aspect,args,tokenizer)
                #     polarity_label=polarity_label.item()
                #     # print(polarity_label)
                #     for (opinion,opinion_logit) in zip(predict_opinion_list,opinion_logits_list):
                #         test_aspect_count+=1
                #         opinion_predict_aspect,opinion_predict_aspect_logits_list=self.predict_aspect_by_opinion(sentence,opinion,args,tokenizer)
                #         if(aspect in opinion_predict_aspect):
                #             opinion_predict_aspect_count+=1
                #             index=opinion_predict_aspect.index(aspect)
                #             predict_result.append((aspect,opinion,polarity_label))
                #         if(args.do_test):
                #             with open("{}_test_logit_{}.txt".format(args.test_logit_out,args.aspect_threshold),"a") as out_fp:
                #                 out_fp.write("{}\naspect:{}\t{}\nopinion:{}\t{}\naspect_by_opinion: {}\t{}\n\n".format(sentence,aspect,aspect_logits,opinion,opinion_logit,list(opinion_predict_aspect),list(opinion_predict_aspect_logits_list)))
                for (aspect,aspect_logits) in zip(predict_aspect_list,aspect_logits_list):
                    predict_opinion_list,polarity_label,opinion_logits_list=self.predict_opinion_by_aspect(sentence,aspect,args,tokenizer)
                    polarity_label=polarity_label.item()
                    # print(polarity_label)
                    for (opinion,opinion_logit) in zip(predict_opinion_list,opinion_logits_list):
                        test_aspect_count+=1
                        if(args.train_binary):
                            predict_result.append((aspect,opinion))
                        else:
                            predict_result.append((aspect,opinion,polarity_label))
                predict_result=set(predict_result)
                
                dual_predict_result=list()
                predict_dual_opinion_list,dual_opinion_logits_list=self.predict_dual_opinion(sentence,args,tokenizer)
                predict_dual_opinion_list=set(predict_dual_opinion_list)
                dual_opinion_pred_num+=len(predict_dual_opinion_list)
                dual_opinion_correct_num+=len(predict_dual_opinion_list&dual_opinion_list)
                dual_predict_result=list()
                for (dual_opinion,dual_opinion_logits) in zip(predict_dual_opinion_list,dual_opinion_logits_list):
                    predict_aspect_list,polarity_label,aspect_logits_list=self.predict_aspect_by_opinion(sentence,dual_opinion,args,tokenizer)
                    polarity_label=polarity_label.item()
                    for (aspect,aspect_logit) in zip(predict_aspect_list,aspect_logits_list):
                        if(args.train_binary):
                            dual_predict_result.append((aspect,dual_opinion))
                        else:
                            dual_predict_result.append((aspect,dual_opinion,polarity_label))
                dual_predict_result=set(dual_predict_result)
                
                predict_result=predict_result&dual_predict_result
                pairs=set(pairs)
                not_in_good= predict_result - pairs
                not_in_predict=pairs-predict_result
                correct=predict_result&pairs
                
                pred_num+=len(predict_result)
                gold_num+=len(pairs)
                correct_num+=len(predict_result&pairs)
                triplet=[list(x) for x in predict_result]
                out_result.append([sentence,triplet,[list(x) for x in pairs],[list(x) for x in not_in_good],[list(x) for x in not_in_predict],[list(x) for x in correct]])
            precision=correct_num/pred_num if pred_num>0 else 0
            recall=correct_num/gold_num
            f1=2*precision*recall/(precision+recall) if precision>0 and recall>0 else 0

            aspect_precision=aspect_correct_num/aspect_pred_num if aspect_pred_num>0 else 0
            aspect_recall=aspect_correct_num/aspect_gold_num if aspect_gold_num>0 else 0
            aspect_f1=2*aspect_precision*aspect_recall/(aspect_recall+aspect_precision) if aspect_precision>0 and aspect_recall>0 else 0
            
            dual_opinion_precision=dual_opinion_correct_num/dual_opinion_pred_num if dual_opinion_pred_num>0 else 0
            dual_opinion_recall=dual_opinion_correct_num/dual_opinion_gold_num if dual_opinion_gold_num>0 else 0
            dual_opinion_f1=2*dual_opinion_precision*dual_opinion_recall/(dual_opinion_recall+dual_opinion_precision) if dual_opinion_precision>0 and dual_opinion_recall>0 else 0

            if(f1>self.best_f1 and args.do_train):
                torch.save({'state_dict': self.model.state_dict(), 'epoch': cur_epoch}, args.save_model_path+"_best.pth")
                self.best_f1=f1
            logging.info("epoch {} | f1 {} | pre {} | recall {} | best f1 {} ".format(cur_epoch,f1,precision,recall,self.best_f1))
            if(args.do_test):
                with open("result/{}_test_result.txt".format(args.data_set),"a",encoding="utf-8") as out_fp:
                    out_fp.write("epoch {} | f1 {} | pre {} | recall {} | best f1 {} \n".format(cur_epoch,f1,precision,recall,self.best_f1))
                with open("result/{}_test_aspect_opinion_result.txt".format(args.data_set),"a",encoding="utf-8") as out_fp:
                    out_fp.write("epoch {} aspect | f1 {} | pre {} | recall {}\n".format(cur_epoch,aspect_f1,aspect_precision,aspect_recall))
                    out_fp.write("epoch {} opinion | f1 {} | pre {} | recall {}\n".format(cur_epoch,dual_opinion_f1,dual_opinion_precision,dual_opinion_recall))
                with open(args.evaluate_out+args.data_set+".txt","w") as out_fp:
                    json.dump(out_result,out_fp,ensure_ascii=False)
            if(args.grid_evaluate):
                if(f1>self.best_f1):
                    self.best_f1=f1
                with open("result.txt","a",encoding="utf-8") as out_fp:
                    out_fp.write("epoch {} | f1 {} | pre {} | recall {} | best f1 {} \n".format(cur_epoch,f1,precision,recall,self.best_f1))


    
    def train(self,train_dataloader,eval_dataloader,args,tokenizer):
        for cur_epoch in range(args.epochs):
            self.train_one_epoch(cur_epoch,train_dataloader,args)
            if(cur_epoch>2):
                self.evaluate(cur_epoch,eval_dataloader,args,tokenizer)
        torch.save({'state_dict': self.model.state_dict(), 'epoch': cur_epoch}, args.save_model_path)
            # self.evaluate(cur_epoch,eval_dataloader,args,tokenizer)
    

def label_data(trainer,test_dataloader,args,tokenizer):
    trainer.model.eval()
    with torch.no_grad():
        label_data=list()
        unlabel_data=list()
        pred_num=0
        gold_num=0
        correct_num=0
        aspect_pred_num=0
        aspect_gold_num=0
        aspect_correct_num=0
        for example in tqdm(test_dataloader):
            
            sentence,aspect_list,pairs,line=example.read_example(True)
            aspect_list=set(aspect_list)
            result=set()
            predict_aspect_span_start_end,predict_aspect_list=trainer.predict_aspect(sentence,args,tokenizer,return_index=True)

            aspect_gold_num+=len(aspect_list)
            aspect_pred_num+=len(predict_aspect_list)
            aspect_correct_num+=len(set(predict_aspect_list)&(aspect_list))
            temp_pair_resutl=set()

            for aspect_span_index,aspect_span in zip(predict_aspect_span_start_end,predict_aspect_list):
                predict_opinion_span_start_end,predict_opinion_list,polarity_label=trainer.predict_opinion_by_aspect(sentence,aspect_span,args,tokenizer,return_index=True)
                polarity_label=polarity_label.item()

                for opinion_span_index,opinion_span in zip(predict_opinion_span_start_end,predict_opinion_list):

                    temp_pair_resutl.add((aspect_span,opinion_span,polarity_label))

                    new_predict_aspect_span_start_end,new_predict_aspect_list=trainer.predict_aspect_by_opinion(sentence,opinion_span,args,tokenizer,return_index=True)
                    if(aspect_span_index in new_predict_aspect_span_start_end):
                        result.add((aspect_span_index[0],aspect_span_index[1],opinion_span_index[0],opinion_span_index[1],polarity_label))
            
            pred_num+=len(temp_pair_resutl)
            gold_num+=len(set(pairs))
            correct_num+=len(set(pairs)&temp_pair_resutl)

            if(len(result)==0):
                unlabel_data.append(line)
            else:
                sentence_list=sentence.split(" ")
                all_token=list()
                for jj in range(len(sentence_list)):
                    all_token=all_token+tokenizer.tokenize(sentence_list[jj])
                new_sentence=" ".join(all_token)
                label=list()
                for aspect_start,aspect_end,opinion_start,opinion_end,polarity_label in result:
                    aspect=list()
                    for index in range(aspect_start,aspect_end):
                        aspect.append(index)
                    opinion=list()
                    for index in range(opinion_start,opinion_end):
                        opinion.append(index)
                    label.append((aspect,opinion,polarity_label))
                label_data.append("{}####{}".format(new_sentence,label))
        precision=correct_num/pred_num if pred_num>0 else 0
        recall=correct_num/gold_num
        f1=2*precision*recall/(precision+recall) if precision>0 and recall>0 else 0

        aspect_precision=aspect_correct_num/aspect_pred_num if aspect_pred_num>0 else 0
        aspect_recall=aspect_correct_num/aspect_gold_num if aspect_gold_num>0 else 0
        aspect_f1=2*aspect_precision*aspect_recall/(aspect_recall+aspect_precision) if aspect_precision>0 and aspect_recall>0 else 0

        logging.info("label | f1 {} | pre {} | recall {} ".format(f1,precision,recall))
        logging.info("label aspect | f1 {} | pre {} | recall {}".format(aspect_f1,aspect_precision,aspect_recall))

    return label_data,unlabel_data
            


def test(trainer,test_dataloader,args,tokenizer):
    all_input_ids,all_pred_span,all_gold_span,all_first_index=trainer.evaluate(0,test_dataloader,args)
    with open("{}_test.txt".format(args.evaluate_out),"w") as out_fp:
        with open("{}_test_wrong.txt".format(args.evaluate_out),"w") as wrong_fp:
            all_count=0
            wrong_count=0
            pred_much=0
            gold_much=0
            pred_gold_all=0
            for input_ids,pred_span,gold_span,first_index in zip(all_input_ids,all_pred_span,all_gold_span,all_first_index):
                all_count+=1
                input_ids=input_ids[:input_ids.index(0)]
                sentence=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
                pred_span_text=list()
                for start,end in pred_span:
                    tokens=input_ids[start+first_index:end+first_index]
                    opinion=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
                    pred_span_text.append((start,end,opinion))
                gold_span_text=list()
                for start,end in gold_span:
                    tokens=input_ids[start+first_index:end+first_index]
                    opinion=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
                    gold_span_text.append((start,end,opinion))
                not_in_pred=list(set(gold_span_text)-set(pred_span_text))
                not_in_gold=list(set(pred_span_text)-set(gold_span_text))
                out_fp.write("{}\t{}\npredict: {}\ngold: {} \n not_in_pred: {}\n not_in_gold: {}\n\n".format(all_count,sentence,pred_span_text,gold_span_text,not_in_pred,not_in_gold))
                if(len(not_in_pred)!=0 or len(not_in_gold) !=0):
                    wrong_count+=1
                    wrong_fp.write("{}\t{}\npredict: {}\ngold: {} \n not_in_pred: {}\n not_in_gold: {}\n\n".format(wrong_count,sentence,pred_span_text,gold_span_text,not_in_pred,not_in_gold))
                    if(len(not_in_pred)==0 and len(not_in_gold)>=0):
                        gold_much+=1
                    elif(len(not_in_pred)>=0 and len(not_in_gold)==0):
                        pred_much+=1
                    else:
                        pred_gold_all+=1
            wrong_fp.write("{}\t{}\t{}\n".format(pred_much,gold_much,pred_gold_all))

                # input_ids=input_ids[:input_ids.index(0)]
                # sentence=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
                # if(predict_start==gold_start and predict_end==gold_end):
                #     tokens=input_ids[gold_start:gold_end]
                #     opinion=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
                #     out_fp.write("{}\t{}\t{}\t{}\n".format(sentence,(str(gold_start)+","+str(gold_end)),(str(predict_start)+","+str(predict_end)),opinion))
                # else:
                #     tokens=input_ids[gold_start:gold_end]
                #     opinion=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
                #     predict_opinion=None
                #     if(predict_end>=predict_start):
                #         tokens=input_ids[predict_start:predict_end]
                #         predict_opinion=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
                #     wrong_fp.write("{}\t{}\t{}\t{}\t{}\n".format(sentence,str(gold_start)+","+str(gold_end),str(predict_start)+","+str(predict_end),opinion,predict_opinion))


    

def main(args):
    tokenizer=BertTokenizer.from_pretrained(args.model_path)
    map_polarity_id_to_token_id(tokenizer)
    train_dataset=MrcAspectOpinionDataset(os.path.join(args.data_dir,args.train_file),tokenizer,args)
    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    eval_dataloader=read_evaluate_example((os.path.join(args.data_dir,args.dev_file)),tokenizer)
    test_dataloader=read_evaluate_example((os.path.join(args.data_dir,args.test_file)),tokenizer)
    trainer=MrcOpinionTrainer(tokenizer,args,len(train_dataset)*args.epochs//args.batch_size)
    if(args.do_train):
        trainer.train(train_dataloader,eval_dataloader,args,tokenizer)
    args.do_train=False
    args.do_test=True
    checkpoint=torch.load(args.save_model_path)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.evaluate(0,test_dataloader,args,tokenizer)
    checkpoint=torch.load(args.save_model_path+"_best.pth")
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.evaluate(1,test_dataloader,args,tokenizer)

    if(args.iter):
        unlabel_dataloader=read_evaluate_example((os.path.join(args.data_dir,args.unlabel_file)),tokenizer)
        # trainer.evaluate(0,unlabel_dataloader,args,tokenizer)
        label_data_result,new_unlabael_data=label_data(trainer,unlabel_dataloader,args,tokenizer)
        return label_data_result,new_unlabael_data

    

# def grid_evaluate(args):
#     tokenizer=BertTokenizer.from_pretrained(args.model_path)
#     train_dataset=MrcAspectOpinionDataset(os.path.join(args.data_dir,args.train_file),tokenizer,args)
#     train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
#     eval_dataloader=read_evaluate_example((os.path.join(args.data_dir,args.dev_file)),tokenizer)
#     test_dataloader=read_evaluate_example((os.path.join(args.data_dir,args.test_file)),tokenizer)
#     trainer=MrcOpinionTrainer(tokenizer,args,len(train_dataset)*args.epochs//args.batch_size)
#     args.do_train=False
#     checkpoint=torch.load(args.save_model_path+"_epoch_19.pth")
#     trainer.model.load_state_dict(checkpoint["state_dict"])
#     for aspect_threshold in range(12,14):
#         # aspect_threshold=aspect_threshold/10
#         for opinion_threshold in range(12,14):
#             args.aspect_threshold=aspect_threshold
#             args.opinion_threshold=opinion_threshold
#             trainer.evaluate("{} {}".format(aspect_threshold,opinion_threshold),eval_dataloader,args,tokenizer)
#             trainer.evaluate("{} {}".format(aspect_threshold,opinion_threshold),test_dataloader,args,tokenizer)

def write_label_data(args,out_dir,label_data_result,new_unlabael_data):
    copyfile(os.path.join(args.data_dir,args.test_file),os.path.join(out_dir,args.test_file))
    copyfile(os.path.join(args.data_dir,args.dev_file),os.path.join(out_dir,args.dev_file))
    copyfile(os.path.join(args.data_dir,args.train_file),os.path.join(out_dir,args.train_file))
    copyfile(os.path.join(args.data_dir,"left.txt"),os.path.join(out_dir,"left.txt"))
    with open(os.path.join(out_dir,args.unlabel_file),"w") as out_fp:
        for line in new_unlabael_data:
            out_fp.write(line+"\n")
    with open(os.path.join(out_dir,args.train_file),"a") as out_fp:
        for line in label_data_result:
            out_fp.write(line+"\n")


def iter(args):
    cur_iter=0
    args.save_model_path_ori=args.save_model_path
    while True:
        args.do_train=True
        args.cur_iter=cur_iter
        args.data_dir=os.path.join(args.toral_data_dir,"iter-{}".format(cur_iter))
        args.save_model_path=args.save_model_path_ori+"_{}.pth".format(cur_iter)
        label_data_result,new_unlabael_data=main(args)
        logging.info("iter-{}：label {}, unlabel {}".format(cur_iter,len(label_data_result),len(new_unlabael_data)))
        if(len(label_data_result)==0 or len(new_unlabael_data)==0):
            break
        out_data_dir=os.path.join(args.toral_data_dir,"iter-{}".format(cur_iter+1))
        if not os.path.exists(out_data_dir):
            os.mkdir(out_data_dir)
        write_label_data(args,out_data_dir,label_data_result,new_unlabael_data)
        cur_iter+=1
        



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--max_length",default=128,type=int)
    parser.add_argument("--max_answer_length",type=int,default=10)
    parser.add_argument("--n_best_size",type=int,default=20)
    parser.add_argument("--threshold",type=float,default=12)
    # parser.add_argument("--aspect_threshold",type=float,default=12)
    # parser.add_argument("--opinion_threshold",type=float,default=12)
    # parser.add_argument("--aspect_by_opinion_threshold",type=float,default=12)
    parser.add_argument("--model_path",type=str,default="bert-base-uncased")
    parser.add_argument("--save_model_path",type=str,required=True)
    parser.add_argument("--lr",type=float,default=5e-5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--batch_size",type=int,default=12)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--device",type=str,default="cuda")
    # parser.add_argument("--data_dir",type=str,default="/home/gdmlab/chentao/aspect_opinion/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res")
    # parser.add_argument("--data_dir",type=str,default="../SemEval-Triplet-data/ASTE-Data-V1-AAAI2020-process")
    parser.add_argument("--data_dir",type=str,default="../SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020")
    parser.add_argument("--toral_data_dir",type=str,default="/home/dell/chentao/mrc-aspcet-opinion/aspect_opinion_dual_semi_triple/data")
    parser.add_argument("--data_set",type=str,default="14res")
    parser.add_argument("--train_file",type=str,default="train_triplets.txt")
    parser.add_argument("--dev_file",type=str,default="dev_triplets.txt")
    parser.add_argument("--test_file",type=str,default="test_triplets.txt")
    parser.add_argument("--unlabel_file",type=str,default="unlabel.txt")
    parser.add_argument("--do_train",action="store_true")
    parser.add_argument("--do_test",action="store_true")
    parser.add_argument("--grid_evaluate",action="store_true")
    parser.add_argument("--train_binary",action="store_true")
    parser.add_argument("--iter",action="store_true")
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--use_pos_tag",action="store_true")

    args=parser.parse_args()
    if(args.seed is not None):
        setup_seed(args.seed)
    args.evaluate_out=os.path.join("evaluate",args.save_model_path)
    args.test_logit_out=os.path.join("test_logits",args.save_model_path)
    args.save_model_path=os.path.join("model",args.save_model_path)
    args.toral_data_dir=os.path.join(args.toral_data_dir,args.data_set)

    logging.info(args)
    if(args.iter):
        iter(args)
    else:
        args.data_dir=os.path.join(args.data_dir,args.data_set)
        main(args)
    # if(args.grid_evaluate):
    #     grid_evaluate(args)
    # else:
    #     main(args)