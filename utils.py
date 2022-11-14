import collections
import re,torch
def cal_metric(all_predict_start,all_predict_end,all_gold_start,all_gold_end):
    total=len(all_predict_end)
    count=0
    # precion=0
    # recall=0
    # clap=0
    # for predict_start,predict_end,gold_start,gold_end in zip(all_predict_start,all_predict_end,all_gold_start,all_gold_end):
    #     precion+=abs(predict_start-predict_end)
    #     recall+=abs(gold_start-gold_end)
    #     predict_set=set(list(range(predict_start,predict_end)))
    #     gold_set=set(list(range(gold_start,gold_end)))
    #     clap+=len(predict_set&gold_set)
    # precion=clap/precion
    # recall=clap/recall
    # f1=2*precion*recall/(precion+recall)
    # return f1
            

    for predict_start,predict_end,gold_start,gold_end in zip(all_predict_start,all_predict_end,all_gold_start,all_gold_end):
        if(predict_start==gold_start and predict_end==gold_end):
            count+=1
    return count/total

def evaluate_metric(pred_spans,gold_spans):
    pred_num=0
    gold_num=0
    correct_num=0
    for cur_pred,cur_gold in zip(pred_spans,gold_spans):
        cur_pred=set(cur_pred)
        cur_gold=set(cur_gold)
        pred_num+=len(cur_pred)
        gold_num+=len(cur_gold)
        correct_num+=len(cur_gold&cur_pred)
    precision=correct_num/pred_num if pred_num >0 else 0
    recall=correct_num/gold_num
    f1=2*precision*recall/(precision+recall) if precision >0 and recall >0 else 0
    return precision,recall,f1


def get_best_index(prob,n_best_size):
    index_and_score = sorted(enumerate(prob), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def iswrap(start_index,end_index,test_start,test_end):
    if(start_index>=test_end or end_index<=test_start):
        return False
    else:
        return True
PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])

def extract_one_by_logits(start_logits,end_logits,sentence_length,threshold,args):
    start_indexes=get_best_index(start_logits,args.n_best_size)
    end_indexes=get_best_index(end_logits,args.n_best_size)
    prelim_predictions_per_feature=list()
    span_start_end=[]

    for start_index in start_indexes:
        for end_index in end_indexes:
            if(start_index>=sentence_length or end_index>sentence_length):
                continue
            if(start_index>=end_index):
                continue
            if(end_index-start_index>args.max_answer_length):
                continue
            start_prob=start_logits[start_index]
            end_prob=end_logits[end_index]
            if(start_prob+end_prob<threshold):
                continue
            prelim_predictions_per_feature.append(PrelimPrediction(start_index=start_index,end_index=end_index,start_logit=start_prob,end_logit=end_prob))
    # 这里要看看end_index和start_index的取值，看看end_index代表的token是否包含在答案里面，然后看看+1还是+0....好像没影响，反正所有样本都加的 1
    prelim_predictions_per_feature = sorted(prelim_predictions_per_feature,key=lambda x: (x.start_logit + x.end_logit - (x.end_index - x.start_index + 1)),reverse=True)

    while len(prelim_predictions_per_feature)>0:
        if(len(span_start_end)>=(args.n_best_size//2)):
            break
        pred_i=prelim_predictions_per_feature[0]
        span_start_end.append((pred_i.start_index,pred_i.end_index))
        new_prelim_predictions_pre_feature=list()
        for ii in range(1,len(prelim_predictions_per_feature)):
            cur_pred_i=prelim_predictions_per_feature[ii]
            if(not iswrap(pred_i.start_index,pred_i.end_index,cur_pred_i.start_index,cur_pred_i.end_index)):
                new_prelim_predictions_pre_feature.append(cur_pred_i)
        prelim_predictions_per_feature=new_prelim_predictions_pre_feature
    return span_start_end

def extract_one_by_global_logits(logits,sentence_length,query_length,args,threshold=0,):
    origin_logits=logits
    logits=torch.gt(logits,threshold)+0
    index=torch.nonzero(logits).cpu().numpy().tolist()
    span_start_end=list()
    sentence_end_index=sentence_length+query_length
    logit_list=list()
    for start,end in index:
        if(start<query_length or end>sentence_end_index):
            continue
        if(start>=end):
            continue
        if((end-start)>8):
            continue
        span_start_end.append((start-query_length,end-query_length))
        logit_list.append(origin_logits[start,end].item())
    return span_start_end,logit_list
    

def extract_all_by_logits(start_prob_all,end_prob_all,sentence_length,args):
    all_pred_span=list()
    for cur_index,(cur_start,cur_end) in enumerate(zip(start_prob_all,end_prob_all)):
        all_pred_span.append(extract_one_by_logits(cur_start,cur_end,sentence_length[cur_index],args.threshold,args))
    return all_pred_span

def convert_span_start_end_to_text(span_start_end,input_ids,tokenizer,fix_index_bias=0):
    result=list()
    for start,end in span_start_end:
        tokens=input_ids[start+fix_index_bias:end+fix_index_bias]
        text=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
        # text=text.replace(" - ","-")
        result.append(text)
    return result