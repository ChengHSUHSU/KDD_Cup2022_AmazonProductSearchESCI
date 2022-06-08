

import numpy as np
from tqdm import tqdm


import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer



 

def build_query2passage5score(query_list=list, 
                              query2data=dict,
                              pd2data=dict,
                              auto_model=None, 
                              auto_trf=None,
                              args=None):
    # init container
    query2passage5score = dict() 
    data_x_infer = []

    # collect passage data
    for query in query_list:
        pdi_list = query2data[query]['all']
        data_x_infer += [[query, pdi] for pdi in pdi_list]

    # init batch_num
    batch_num = int(len(data_x_infer) / args.batch_size) + 1
        
    # main - infer
    for i in tqdm(range(batch_num)):
        data_x_batch = data_x_infer[i*args.batch_size : (i+1)*args.batch_size]
        if len(data_x_batch) != 0:
            data_text_x_batch, sent_length = convert_q_pdi_to_q_sent_feature(q_pdi_list=data_x_batch,
                                                                             pd2data=pd2data,
                                                                             eval_mode=True,
                                                                             args=args)
                
            score_list = AutoCrossEncoder_feature(head_tail_list=data_text_x_batch, 
                                                  auto_model=auto_model, 
                                                  auto_trf=auto_trf, 
                                                  sent_length=sent_length).tolist()
           
            for j, query_sent_feature in enumerate(data_text_x_batch):
                score = score_list[j][0]
                query = query_sent_feature[0]
                sent_feature = query_sent_feature[1:]
                sent_feature = '. '.join(sent_feature)
                query_, pdi = data_x_batch[j]
                if query not in query2passage5score:
                    query2passage5score[query] = {'mapping_score' : [], 'mapping_entity' : []}
                query2passage5score[query]['mapping_score'].append([pdi, score])
                query2passage5score[query]['mapping_entity'].append(pdi)
    return query2passage5score




def AutoCrossEncoder_feature(head_tail_list=list, auto_model=None, auto_trf=None, sent_length=list):
    bert_input = auto_trf.convert_batch_sent_to_bert_input(batch_sent=head_tail_list, sent_length=sent_length)
    bert_input = auto_trf.transform_bert_input_into_tensor(bert_input=bert_input)
    logits = auto_model(**bert_input).logits
    return logits 




def convert_q_pdi_to_q_sent_feature(q_pdi_list=list, pd2data=dict, eval_mode=False, args=None):
    # init
    q_sent_feature_list = []
    
    if args.contractive_loss is False or eval_mode is True:  
        for element in q_pdi_list:
            query, pdi = element[0], element[1]
            sent_feature, sent_length = convert_pd2sent_feature(pdi=pdi, 
                                                                pd2data=pd2data, 
                                                                args=args)
            q_sent_feature_list.append([query] + sent_feature)
    else:
        for query, pdi, pos_pdi, neg_pdi in q_pdi_list:
            sent_feature, sent_length = convert_pd2sent_feature(pdi=pdi, 
                                                                pd2data=pd2data, 
                                                                args=args)
            pos_sent_feature, sent_length = convert_pd2sent_feature(pdi=pos_pdi, 
                                                                    pd2data=pd2data, 
                                                                    args=args)
            neg_sent_feature, sent_length = convert_pd2sent_feature(pdi=neg_pdi, 
                                                                    pd2data=pd2data, 
                                                                    args=args)
            q_sent_feature_list.append([query] + sent_feature + pos_sent_feature + neg_sent_feature)       
    # add max query length to sent_length
    sent_length = [args.max_query_length] + sent_length
    return q_sent_feature_list, sent_length




def convert_pd2sent_feature(pdi=int, pd2data=dict, args=None):
    # collect sent all feature
    product_title = pd2data[pdi]['product_title']
    super_sents = pd2data[pdi]['super_sents']

    # determine selected feature
    sent_feature = []
    sent_length = []
    for (feature_bool, sent)  in [(args.max_title_length, product_title),
                                  (args.max_super_sents_length, super_sents)]:
        if feature_bool is not None:
            sent_feature.append(sent)
            sent_length.append(feature_bool)
    return sent_feature, sent_length




def calculate_eval_score(passage_pd4score=list, query_data=dict): 
    # init 
    y_true, y_score = [], []
    
    # build_pd2gain
    pd2gain = build_pd2gain(query_data['data'])

    # sort passage_pd4score by score
    passage_pd4score = sorted(passage_pd4score, reverse=True, key=lambda x:x[1]) 

    # build y_true, y_score
    for passage_pd, score in passage_pd4score:
        y_true.append(pd2gain[passage_pd])
        y_score.append(score)
    return y_true, y_score




def build_pd2gain(data_list):
    pd2gain = dict()
    for data in data_list:
        product_new_id = data['product_new_id']
        gain = data['gain']
        pd2gain[product_new_id] = gain
    return pd2gain




def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best




def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])


    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)








class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_list: list,
        block_size: int
    ):
        self.examples = []
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        for text in tqdm(text_list):
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)[:block_size])
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)



def build_corpus(pd2data=dict):
    corpus = list()
    pdi_list = list(pd2data.keys())
    for pdi in tqdm(pdi_list):
        product_title = pd2data[pdi]['product_title']
        product_bullet_point = pd2data[pdi]['product_bullet_point']
        if product_bullet_point != 'Empty':
            text = product_title + ' .' +product_bullet_point
            corpus.append(text)
    return corpus






def build_submit_result(query2test_data=dict, pd2data=dict, auto_model=None, auto_trf=None, args=None):
    # init
    query_list = list(query2test_data.keys())
    submit_dat = {'product_id' : [], 'query_id' : []}
    # main
    with torch.no_grad():
        query2passage_pd5score = build_query2passage5score(query_list=query_list, 
                                                           query2data=query2test_data,
                                                           pd2data=pd2data,
                                                           auto_model=auto_model, 
                                                           auto_trf=auto_trf,
                                                           args=args)
    for query in tqdm(query_list):
        # get query_id
        query_id = query2test_data[query]['query_id']

        # build passage_pd4score
        passage_pd5score = query2passage_pd5score[query]
        passage_pd4score = passage_pd5score['mapping_score'][:]
        
        # replace product_new_id with product_id
        passage_pd4score_update = [[pd2data[pdi]['product_id'], score] for pdi, score in passage_pd4score]
        
        # sort by score
        passage_pd4score_update = sorted(passage_pd4score_update, reverse=True, key=lambda x:x[1])

        # append to submit_dat
        for product_id, score in passage_pd4score_update:
            submit_dat['product_id'].append(product_id)
            submit_dat['query_id'].append(query_id)
    return submit_dat
 
    
    
