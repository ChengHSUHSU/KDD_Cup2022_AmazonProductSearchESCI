



import yaml
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sknetwork.ranking import PageRank
import networkx as nx
from rank_bm25 import BM25Okapi













def load_config(path=str):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg





def convert_q_pdi_to_q_sent_feature(q_pdi_list=list, pd2data=dict, eval_mode=False, args=None):
    # init
    q_sent_feature_list = []
    progress_step = 0
    progress_line = len(q_pdi_list) 
    progress_bar = int(progress_line / 5)
    for i, (query, pdi) in enumerate(q_pdi_list):
        sent_feature, sent_length = convert_pd2sent_feature(q=query,
                                                            pdi=pdi, 
                                                            pd2data=pd2data, 
                                                            args=args)
        q_sent_feature_list.append([query] + sent_feature)
        try:
            if i % progress_bar == 0 and progress_line > 256:
                print('progress_step ({}) :'.format(str(progress_step) + '/5'))
                progress_step += 1
        except:
            pass
    # add max query length to sent_length
    max_query_length = args.model_cfg['max_query_length']
    sent_length = [max_query_length] + sent_length

    return q_sent_feature_list, sent_length




def convert_pdi_pdi_to_sent_feature(q_pdi_pdi_list=list, pd2data=dict, args=None):
    # init parameter
    max_title_length = args.model_cfg['max_title_length']

    # init qpp_sent_feature_list
    qpp_sent_feature_list = []
    
    # build q2p2sent_feature
    q2p2sent_feature = dict()
    for q, pl, pr in tqdm(q_pdi_pdi_list):
        if q not in q2p2sent_feature:
            q2p2sent_feature[q] = dict()
        for p in [pl, pr]:
            if p not in q2p2sent_feature[q]:
                p_sent_feature, sent_length = convert_pd2sent_feature(q=q,
                                                                      pdi=p, 
                                                                      pd2data=pd2data, 
                                                                      args=args)
                q2p2sent_feature[q][p] = [p_sent_feature, sent_length]
    
    # add query, product1, product2 as sent_feature
    for q, pl, pr in q_pdi_pdi_list:
        p1_sent_feature, sent_length = q2p2sent_feature[q][pl]
        p2_sent_feature, sent_length = q2p2sent_feature[q][pr]
        qpp_sent_feature_list.append([q] + p1_sent_feature + p2_sent_feature)

    # add max_query_length, max_title_length to sent_length
    sent_length = [max_query_length] + [max_title_length] + [max_title_length]

    return qpp_sent_feature_list, sent_length





def convert_pd2sent_feature(q=None, pdi=int, pd2data=dict, args=None):
    # init parameter
    max_title_length = args.model_cfg['max_title_length']
    use_additional_pdfeature = args.model_cfg['use_additional_pdfeature']
    additional_pdfeature_locale = args.model_cfg['additional_pdfeature_locale']

    # product feature
    product_title = pd2data[pdi]['product_title']
    product_locale = pd2data[pdi]['product_locale']
    product_bullet_point = pd2data[pdi]['product_bullet_point'] 
    product_description = pd2data[pdi]['product_description'] 
    
    if use_additional_pdfeature is True and q is not None and product_locale in additional_pdfeature_locale:
        # concat bullet_point, description
        text_feature = product_bullet_point.replace('\n', '. ')  + '. ' + product_description
        
        # extract keyword algorithm
        #ksng_info = KeySentNgram_algo(query=q, title=product_title, text_feature=text_feature, ngram=5, test=False)
        #product_title_update = ksng_info['title']  +'. '.join(ksng_info['keysent_ngram'])
        product_title_update = KeySentNgram_algo_new(query=q, product_title=product_title, product_documents=text_feature, mode='BM25', args=args)
        product_title_update = '. '.join(product_title_update)
    else:
        if product_bullet_point != 'Empty':
            af_backups = product_bullet_point.split('\n')
            product_title_update = product_title + '. ' + random.sample(af_backups, 1)[0]
        else:
            product_title_update = product_title

    # output
    sent_feature = [product_title_update]
    sent_length = [max_title_length]
    return sent_feature, sent_length




def KeySentNgram_algo(query=str, title=str, text_feature=str, ngram=5, test=False):
    # init keysent_ngram
    keysent_ngram = []
    
    # build words_query
    words_query = query.replace(',', ' ').split()
    
    # build words_title
    words_title = title.replace(',', ' ').split()
    
    # build words_feature
    words_feature = text_feature.replace(',', ' ').split()

    # build first_word2words_query
    first_word2words_query = build_first_word2words(words=words_query)
    
    # build first_word2words_title
    first_word2words_title = build_first_word2words(words=words_title)
    
    # build first_word2words_feature
    first_word2words_feature = build_first_word2words(words=words_feature)
    if test is True:
        print(first_word2words_feature)
    
    
    # main
    for word_q in words_query:
        word_q = word_q.lower()
        if len(word_q) >= 2:
            if test is True:
                print('word_q : ', word_q)
            # init first word query
            fw_word_q_ord = str(ord(word_q[0])) + '-' + str(ord(word_q[1]))
            # init best token
            best_match_word = None
            best_score = 0
            best_idx = None
            # check the word is in title or not
            if fw_word_q_ord in first_word2words_title:
                match_words_title = first_word2words_title[fw_word_q_ord]
                for m_word_t, idx in match_words_title:
                    score = longest_common_substring(target_w=word_q, source_w=m_word_t)
                    if score >= 0.6 and score > best_score:
                        best_score = score
                        best_match_word = m_word_t
                if test is True:
                    print(123)
                
            if best_match_word is None:
                if fw_word_q_ord in first_word2words_feature:
                    if test is True:
                        print(789)
                    match_words_feature = first_word2words_feature[fw_word_q_ord]
                    if test is True:
                        print(match_words_feature)
                    for m_word_f, idx in match_words_feature:
                        score = longest_common_substring(target_w=word_q, source_w=m_word_f)
                        if score >= 0.6 and score > best_score:
                            best_score = score
                            best_match_word = m_word_f
                            best_idx = idx
                        if test is True:
                            print('best_match_word : ', best_match_word)
                            print('best_idx : ', best_idx)
                            print('best_score : ', best_score)
                    if best_match_word is not None:
                        keysent_ngram.append(best_idx)
                if test is True:
                    print(456)
        if test is True:
            print('-------')
    # expand best_idx
    if test is True:
        print(keysent_ngram)
    keysent_ngram = flatten_idx(idx_List=keysent_ngram, ngram=ngram)
    
    keysent_ngram = [' '.join(words_feature[l: u+1]) for l, u in keysent_ngram]

    # output
    output = {
              'query' : query,
              'title' : title,
              'keysent_ngram' : keysent_ngram
              }
    return output




def build_first_word2words(words=list):
    # init 
    first_word_ord2words= dict()
    # main
    for i in range(len(words)):
        w = words[i]
        w = w.lower()
        if len(w) >= 2:
            first_word_ord = str(ord(w[0])) + '-' + str(ord(w[1]))
            if first_word_ord not in first_word_ord2words:
                first_word_ord2words[first_word_ord] = []
            first_word_ord2words[first_word_ord].append([w, i])
    return first_word_ord2words




def longest_common_substring(target_w=str, source_w=str):
    # find the length of the strings
    m = len(target_w)
    n = len(source_w)
    output = 0
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif target_w[i-1] == source_w[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = 0
            if output < L[i][j]:
                output = L[i][j]
    score = output / len(target_w)
    return score




def flatten_idx(idx_List=list, ngram=int):
    # init container
    flatten_idx_List = []
    # sort idx_List
    idx_List = sorted(idx_List)
    # 
    for idx in idx_List:
        upper = idx + ngram
        lower = idx - ngram
        if lower < 0:
            lower = 0
        if len(flatten_idx_List) == 0:
            flatten_idx_List.append([lower, upper])
        else:
            prev_lower = flatten_idx_List[-1][0]
            prev_upper = flatten_idx_List[-1][1]
            if prev_upper > lower:
                flatten_idx_List[-1][1] = upper
            else:
                flatten_idx_List.append([lower, upper])
    return flatten_idx_List



 
def evaluation(query_list=list, 
               query2data=dict, 
               pd2data=dict, 
               auto_model=None, 
               auto_trf=None, 
               args=None, 
               category='Train'):
    # init parameter
    use_mixed_model = args.model_cfg['use_mixed_model']
    target_query_locale = args.data_process_cfg['target_query_locale']
    model_save_path = args.model_cfg['model_save_path']
    save_training_info = args.model_cfg['save_training_info']
    foolish_threshold = args.model_cfg['foolish_threshold']

    # init container
    ndcg_avg_score = []
    query_locale2query2score = {'us' : dict(), 'es' : dict(), 'jp' : dict()}
    query_locale2query2ndcg_matrix = {'us' : dict(), 'es' : dict(), 'jp' : dict()}
    query2ndcg_score = dict()
    query2pdi2gain_data = dict()
    query_locale2ndcg_avg_score = {'us' : [], 'es' : [], 'jp' : []}
    query2ndcg_matrix = dict()
    query_locale2ndcg_matrix =  {
                                 'us' : init_ndcg_matrix(), 
                                 'es' : init_ndcg_matrix(), 
                                 'jp' : init_ndcg_matrix()
                                }
    # main
    with torch.no_grad():
        # build query2passage_pd5score
        if use_mixed_model is False:
            query2passage_pd5score, training_info = build_query2passage5score(query_list=query_list, 
                                                                              query2data=query2data,
                                                                              pd2data=pd2data,
                                                                              auto_model=auto_model, 
                                                                              auto_trf=auto_trf,
                                                                              args=args)
        else:
            query2passage_pd5score = build_query2passage5score_mixed(query_list=query_list, 
                                                                     query2data=query2data,
                                                                     pd2data=pd2data,
                                                                     auto_model=auto_model, 
                                                                     auto_trf=auto_trf,
                                                                     args=args)
        # for each query, calculate ndcg score
        for query in tqdm(query_list):
            passage_pd5score = query2passage_pd5score[query]
            passage_pd4score = passage_pd5score['mapping_score'][:]

            # build_pd2gain
            pd2gain = build_pd2gain(query2data[query]['data'])
            
            # query_locale
            query_locale = query2data[query]['locale']
            
            # calculate ndcg score
            y_true, y_score = calculate_eval_score(passage_pd4score=passage_pd4score, 
                                                   pd2gain=pd2gain)

            ndcg_score_value = ndcg_score(y_true=y_true, 
                                          y_score=y_score, 
                                          k=len(y_score), 
                                          gains="exponential")
            # check ndcg_score_value is valid
            if pd.isna(ndcg_score_value) is False:
                # overall
                ndcg_avg_score.append(ndcg_score_value)
                
                # specific region
                query_locale2ndcg_avg_score[query_locale].append(ndcg_score_value)
                
                # ndcg_matrix
                query2ndcg_matrix = calculate_ndcg_matrix(query=query, 
                                                          pd2gain=pd2gain, 
                                                          passage_pd4score=passage_pd4score, 
                                                          query2ndcg_matrix=query2ndcg_matrix)
                for label_t in ['E', 'S', 'C', 'I']:
                    for label_s in ['E', 'S', 'C', 'I']:
                        ndcg = query2ndcg_matrix[query][label_t][label_s]
                        if ndcg is not None and pd.isna(ndcg) is False:
                            query_locale2ndcg_matrix[query_locale][label_t][label_s].append(ndcg)
                
                # query_locale2query2ndcg_matrix
                if query in query2ndcg_matrix:
                    query_locale2query2ndcg_matrix[query_locale][query] = query2ndcg_matrix[query]
                else:
                    add_log_record(message='[Warning] : query not in query2ndcg_matrix', args=args)
                
                # query_locale2query2score
                query_locale2query2score[query_locale][query] = ndcg_score_value
            else:
                print('[Warning] : ndcg_score_value is nan.')
                add_log_record(message='[Warning] : ndcg_score_value is nan.', args=args)

        # calculate evaluation metric
        count = len(ndcg_avg_score)
        ndcg_avg = sum(ndcg_avg_score) / len(ndcg_avg_score)
        ndcg_std = ((sum([(sc - ndcg_avg) ** 2 for sc in ndcg_avg_score])) ** (1/2)) / count
        
        # add log
        add_log_record(message='count ({}) : '.format(category) + str(count), args=args)
        add_log_record(message='ndcg_avg ({}) : '.format(category) + str(ndcg_avg), args=args)
        add_log_record(message='ndcg_std ({}) : '.format(category) + str(ndcg_std), args=args)

        # calculate specific evaluation metric
        for locale in target_query_locale:
            spr_ndcg_avg_score = query_locale2ndcg_avg_score[locale]
            if len(spr_ndcg_avg_score) != 0:
                spr_count = len(spr_ndcg_avg_score)
                spr_ndcg_avg = sum(spr_ndcg_avg_score) / len(spr_ndcg_avg_score)
                spr_ndcg_std = ((sum([(sc - spr_ndcg_avg) ** 2 for sc in spr_ndcg_avg_score])) ** (1/2)) / spr_count
                
                # add log
                add_log_record(message='specific region count ({}) ({}) : '.format(category, locale) + str(spr_count), args=args)
                add_log_record(message='specific region ndcg_avg ({}) ({}) : '.format(category, locale) + str(spr_ndcg_avg), args=args)
                add_log_record(message='specific region ndcg_std ({}) ({}) : '.format(category, locale) + str(spr_ndcg_std), args=args)
  
                # take avg for each label pair
                for label_t in ['E', 'S', 'C', 'I']:
                    for label_s in ['E', 'S', 'C', 'I']:
                        ndcg_list = query_locale2ndcg_matrix[locale][label_t][label_s]
                        if ndcg_list is not None and  len(ndcg_list) > 0:
                            query_locale2ndcg_matrix[locale][label_t][label_s] = sum(ndcg_list) / len(ndcg_list)
                dat = pd.DataFrame(query_locale2ndcg_matrix[locale])
                
                # add log
                add_log_record(message='specific region ndcg_matrix ({}) ({}) : '.format(category, locale), args=args)
                add_log_record(message='\n'+dat.to_string(), args=args)
        
        # update training_info (about foolish)
        foolish_vals = []
        train_x = training_info['train_x']
        for query, pdi in train_x:
            # query_locale
            query_locale = query2data[query]['locale']
            if query_locale2query2score[query_locale][query] < foolish_threshold:
                foolish_vals.append(1)
            else:
                foolish_vals.append(0)
        training_info['foolish'] = foolish_vals

        # save query_ndcg for train_data
        if save_training_info is True and category == 'Train':
            path_pkl = model_save_path + '_data.pkl'
            data = {
                    'training_info' : training_info,
                    'query_locale2query2score' : query_locale2query2score,
                    'query_locale2query2ndcg_matrix' : query_locale2query2ndcg_matrix
                   } 
            with open(path_pkl, "wb") as f:
                pickle.dump(data, f)





def init_ndcg_matrix():
    label_list = ['E', 'S', 'C', 'I']
    ndcg_matrix = {'E':{}, 'S':{}, 'C':{}, 'I':{}}
    for label_t in label_list:
        for label_s in label_list:
            if label_t != label_s:
                ndcg_matrix[label_t][label_s] = []
            else:
                ndcg_matrix[label_t][label_s] = None
    return ndcg_matrix





 

def build_pd2gain(data_list):
    pd2gain = dict()
    for data in data_list:
        product_new_id = data['product_new_id']
        gain = data['gain']
        pd2gain[product_new_id] = gain
    return pd2gain



def calculate_eval_score(passage_pd4score=list, pd2gain=dict): 
    # init 
    y_true, y_score = [], []

    # sort passage_pd4score by score
    passage_pd4score = sorted(passage_pd4score, reverse=True, key=lambda x:x[1]) 

    # build y_true, y_score
    for passage_pd, score in passage_pd4score:
        y_true.append(pd2gain[passage_pd])
        y_score.append(score)
    return y_true, y_score




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





def calculate_ndcg_matrix(query=str, pd2gain=dict, passage_pd4score=dict, query2ndcg_matrix=dict):
    ''' 
    E -> E, S, C, I
    S -> E, S, , ...
    '''
    gain2label = {
                  1.0 : 'E',
                  0.1 : 'S',
                  0.01: 'C',
                  0.0 : 'I'
                 }
    label2gain = {
                  'E' : 1.0,
                  'S' : 0.1,
                  'C': 0.01,
                  'I' : 0.0
                 }
    label_list = ['E', 'S', 'C', 'I']
    query2ndcg_matrix[query] = {'E':{}, 'S':{}, 'C':{}, 'I':{}}
    # init
    for label_t in label_list:
        for label_s in label_list:
            query2ndcg_matrix[query][label_t][label_s] = None

    # sort passage_pd4score by score
    passage_pd4score = sorted(passage_pd4score, reverse=True, key=lambda x:x[1]) 

    # build label2idxs
    label2idxs = {'E' : [], 'S' : [], 'C' : [], 'I' : []}
    for idx, (passage_pd, score) in enumerate(passage_pd4score):
        gain = pd2gain[passage_pd]
        label = gain2label[gain]
        label2idxs[label].append(idx)
    
    # for each label_target, label_source, attach confusion_matrix_onehot_score
    for label_t in label_list:
        idxs_t = label2idxs[label_t]
        for label_s in label_list:
            idxs_s = label2idxs[label_s]
            if label_t != label_s and len(idxs_s) != 0 and len(idxs_t) != 0:
                gain_t = label2gain[label_t]
                gain_s = label2gain[label_s]
                if gain_t > gain_s:
                    idxs_t4score = [[idx,1] for idx in idxs_t]
                    idxs_s4score = [[idx,0] for idx in idxs_s]
                    gt_score = [1 for _ in range(len(idxs_t4score))] + [0 for _ in range(len(idxs_s4score))] 
                else:
                    idxs_t4score = [[idx,0] for idx in idxs_t]
                    idxs_s4score = [[idx,1] for idx in idxs_s]
                    gt_score = [1 for _ in range(len(idxs_s4score))] + [0 for _ in range(len(idxs_t4score))] 
                idxs4score = idxs_t4score + idxs_s4score
                idxs4score = sorted(idxs4score)
                cmo_score = [cmo for (idx, cmo) in idxs4score]
                # calculate ndgc
                ndcg_score_value = ndcg_score(y_true=gt_score, 
                                              y_score=cmo_score, 
                                              k=len(gt_score), 
                                              gains="exponential")
            else:
                ndcg_score_value = None
            # put ndcg_score_value into target_label, source_label
            query2ndcg_matrix[query][label_t][label_s] = ndcg_score_value
    return query2ndcg_matrix






def build_query2passage5score(query_list=list, 
                              query2data=dict,
                              pd2data=dict,
                              auto_model=None, 
                              auto_trf=None,
                              args=None):
    # init parameter
    batch_size = args.model_cfg['batch_size']
    use_classfier = args.model_cfg['use_classfier']

    # init container
    query2passage5score = dict() 
    data_x_infer = []
    data_y_infer = []
    training_info = {'train_x' : [], 'train_y' : [], 'logits' : [], 'foolish' : []}

    # gain2label_name (hard code)
    gain2label_name = {1.0 : 'E', 0.1 : 'S', 0.01 : 'C', 0.0 : 'I'}

    # collect passage data
    for query in query_list:
        data_list = query2data[query]['data']
        data_x_infer += [[query, data['product_new_id']] for data in data_list]
        data_y_infer += [gain2label_name[data['gain']] for data in data_list]

    # init batch_num
    batch_num = int(len(data_x_infer) / batch_size) + 1

        
    # main - infer
    for i in tqdm(range(batch_num)):
        data_x_batch = data_x_infer[i*batch_size : (i+1)*batch_size]
        data_y_batch = data_y_infer[i*batch_size : (i+1)*batch_size]
        if len(data_x_batch) != 0:
            data_text_x_batch, sent_length = convert_q_pdi_to_q_sent_feature(q_pdi_list=data_x_batch,
                                                                             pd2data=pd2data,
                                                                             eval_mode=True,
                                                                             args=args)

                
            score_list, logits_ent = AutoCrossEncoder_feature(head_tail_list=data_text_x_batch,
                                                              auto_model=auto_model, 
                                                              auto_trf=auto_trf, 
                                                              sent_length=sent_length,
                                                              args=args,
                                                              use_classfier=use_classfier)
            score_list = score_list.tolist()
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
            # build foolish_vals (when teacher perf is bad, we dont use it)

            # save training info
            training_info['train_x'] += data_x_batch
            training_info['train_y'] += data_y_batch
            training_info['logits'] += logits_ent
    return query2passage5score, training_info




def AutoCrossEncoder_feature(head_tail_list=list, 
                             auto_model=None, 
                             auto_trf=None, 
                             sent_length=list, 
                             args=None, 
                             use_classfier=False):

    bert_input = auto_trf.convert_batch_sent_to_bert_input(batch_sent=head_tail_list, sent_length=sent_length)
    bert_input = auto_trf.transform_bert_input_into_tensor(bert_input=bert_input)
    if use_classfier is False:
        logits = auto_model(**bert_input).logits
        logits_ent = logits.tolist()[:]
    else:
        num_labels = args.model_cfg['num_labels']
        logits = auto_model(**bert_input).logits
        logits_ent = logits.tolist()[:]
        logits = torch.sigmoid(logits)
        pred_labels = np.argmax(logits.tolist(), axis=1)
        pred_labels_onehot = np.zeros((pred_labels.size, num_labels))
        pred_labels_onehot[np.arange(pred_labels.size), pred_labels] = 1
        score_array = torch.sum(logits.cpu() * pred_labels_onehot, axis=1).numpy() + (num_labels-1 - pred_labels)
        logits = score_array.reshape(-1, 1)
    return logits, logits_ent





def build_query2passage5score_mixed(query_list=list, 
                              query2data=dict,
                              pd2data=dict,
                              auto_model=None, 
                              auto_trf=None,
                              args=None):
    # init parameter
    batch_size = args.model_cfg['batch_size']
    model_info = args.model_cfg['model_info']

    # init container
    query2passage5score = dict() 
    locale2data_x_infer = {'us' : [], 'es' : [], 'jp' : []}
    training_info = {'train_x' : [], 'train_y' : [], 'logits' : []}

    # init locale2model_name2qps
    locale2model_name2qps = {'us' : dict(), 'es' : dict(), 'jp' : dict()}
    for locale in ['us', 'es', 'jp']:
        model_name_list = model_info[locale]
        for model_name, weight in model_name_list:
            locale2model_name2qps[locale][model_name] = dict()

    # collect data_x_infer for each locales
    for query in query_list:
        pdi_list = query2data[query]['all']
        query_locale = query2data[query]['locale']
        locale2data_x_infer[query_locale] += [[query, pdi] for pdi in pdi_list]


    # collect data to locale2model_name2qps
    for locale in list(locale2data_x_infer.keys()):
        
        # init data_x_infer, batch_num
        data_x_infer = locale2data_x_infer[locale]
        batch_num = int(len(data_x_infer) / batch_size) + 1

        # lauch model
        model_name_list = model_info[locale]
        for model_name, weight in model_name_list:
            if model_name in auto_model:
                model_ = auto_model[model_name]
                trf_ = auto_trf[model_name]
                if 'classifier' in model_name:
                    use_classfier = True
                else:
                    use_classfier = False

                # main - infer
                for i in tqdm(range(batch_num)):
                    data_x_batch = data_x_infer[i*batch_size : (i+1)*batch_size]
                    if len(data_x_batch) != 0:
                        # build batch x
                        data_text_x_batch, sent_length = convert_q_pdi_to_q_sent_feature(q_pdi_list=data_x_batch,
                                                                                        pd2data=pd2data,
                                                                                        eval_mode=True,
                                                                                        args=args)
                        # infer score    
                        score_list, _ = AutoCrossEncoder_feature(head_tail_list=data_text_x_batch, 
                                                                 auto_model=model_, 
                                                                 auto_trf=trf_, 
                                                                 sent_length=sent_length,
                                                                 args=args,
                                                                 use_classfier=use_classfier)
                        score_list = score_list.tolist()
                        # collect query-product-score data
                        for j, query_sent_feature in enumerate(data_text_x_batch):
                            score = score_list[j][0]
                            query = query_sent_feature[0]
                            query_, pdi = data_x_batch[j]
                            if query not in locale2model_name2qps[locale][model_name]:
                                locale2model_name2qps[locale][model_name][query] = dict()
                            locale2model_name2qps[locale][model_name][query][pdi] = score
    
    # calculate max, min for each locale-model_name
    for locale in list(locale2data_x_infer.keys()):
        model_name2qps = locale2model_name2qps[locale]
        for model_name, weight in model_info[locale]:
            score_list = []
            max_score = 1
            min_score = 2
            qps = model_name2qps[model_name]
            q_list = list(qps.keys())
            for q in q_list:
                p_list = list(qps[q].keys())
                for p in p_list:
                    score_list.append(qps[q][p])
            if len(score_list) != 0:
                max_score = max(score_list)
                min_score = min(score_list)
            if max_score - min_score == 0:
                print('[Warning] : max_score - min_score == 0')
                print('locale : ', locale)
                print('model_name : ', model_name)
            elif max_score - min_score < 0:
                print('[Warning] : there is no score data')
                print('locale : ', locale)
                print('model_name : ', model_name)
            else:
                # update qps
                for q in q_list:
                    p_list = list(qps[q].keys())
                    for p in p_list:
                        new_s = (qps[q][p] - min_score) / (max_score - min_score)
                        qps[q][p] = new_s
                # update qps
                locale2model_name2qps[locale][model_name] = qps

    # build query2passage5score
    for query in query_list:
        # build basic container
        if query not in query2passage5score:
            query2passage5score[query] = {'mapping_score' : [], 'mapping_entity' : []}
        # query_locale
        query_locale = query2data[query]['locale']
        # init p2s_list
        p2s_list = dict()
        pdi_list = query2data[query]['all']
        for p in pdi_list:
            p2s_list[p] = []
        # collect p2s_list
        for model_name, weight in model_info[query_locale]:
            p2s = locale2model_name2qps[query_locale][model_name][query]
            for p in pdi_list:
                s = p2s[p]
                p2s_list[p].append(s * weight)
        # calculate avg_score
        for pdi in pdi_list:
            avg_score = sum(p2s_list[pdi]) #/ len(p2s_list[pdi])
            query2passage5score[query]['mapping_score'].append([pdi, avg_score])
            query2passage5score[query]['mapping_entity'].append(pdi)

    return query2passage5score




def setup_logger(date, logger_name, log_file):
    import logging
    level=logging.INFO
    try:
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter('{} : %(message)s'.format(date))
        fileHandler = logging.FileHandler(log_file, mode='a')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fileHandler)
        l.addHandler(streamHandler)
        l.removeHandler(streamHandler)
    except Exception as error_message:
        print(error_message)



def add_log_record(message, args=None):
    import logging
    from datetime import datetime
    # init parameter
    model_save_path = args.model_cfg['model_save_path']
    # add log
    try:
        now_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now_date = now_datetime.split()[0]
        setup_logger(now_datetime, 'log', 'save_log/{}.log'.format(model_save_path.replace('/', '-')))
        log = logging.getLogger('log')

        streamhandler = logging.StreamHandler()
        log.addHandler(streamhandler)
        log.info(message)
        log.removeHandler(streamhandler)

    except Exception as error_message:
        print('error_message---------------')
        print(error_message)






def sync_rel_matrix_into_graph(node_list=list, rel_matrix=dict, graph=None):
    for i in range(len(node_list)):
        node_i = node_list[i]
        for j in range(len(node_list)):
            node_j = node_list[j]
            if i != j:
                weight = rel_matrix[node_i][node_j]
                graph = build_digraph(graph=graph, 
                                      edges=[(i, j, weight)], 
                                      mode='add_weighted_edges')
    return graph



def ExtractSentByBM25(target_texts=list, source_texts=list):
    extracted_sents = []
    source_sents = [source_sent.split() for source_sent in source_texts]
    bm25 = BM25Okapi(source_sents)
    for target_sent in target_texts:
        source_sents_score = list(bm25.get_scores(target_sent.split()))
        source_sents4score = [[' '.join(sent), source_sents_score[i]] for i, sent in enumerate(source_sents)]
        source_sents4score = sorted(source_sents4score, reverse=True, key=lambda x:x[1])
        for sent, score in source_sents4score:
            if sent not in extracted_sents:
                extracted_sents.append(sent)
                break
    return extracted_sents





def build_adjacency_matrix(graph):
    row = []
    col = []
    data = []
    nodes = graph.nodes()
    length = len(nodes)
    for current_node, _ in nodes.items():
        neighbors_sum = sum(graph.get_edge_data(current_node, neighbor)['weight'] for neighbor in graph.neighbors(current_node))
        #neighbors_sum = 1.0
        for neighbor_node, _ in nodes.items():
            edge_weight = graph.get_edge_data(current_node, neighbor_node)
            if edge_weight is not None:
                edge_weight = float(edge_weight['weight'])
            else:
                edge_weight = 0.0
            if edge_weight != 0:
                row.append(current_node)
                col.append(neighbor_node)
                data.append(edge_weight / neighbors_sum)
    return csr_matrix((data,(row,col)), shape=(length,length))



def build_digraph(graph=None, edges=list, mode=None):
    '''
    ex. 
    [(1, 0, 0.5),
     (2, 0, 0.1),
     (0, 1, 0.1),
     (2, 1, 0.1)]
    '''
    if mode == 'init':
        graph = nx.DiGraph()
    elif mode == 'add_weighted_edges':
        graph.add_weighted_edges_from(edges)
    elif mode == 'clone':
        graph_cp = graph.copy()
        return graph_cp
    return graph



def build_rel_matrix(target_texts=list, source_texts=list, metric_mode=None):
    # init 
    rel_matrix = dict()
    all_scores = []
    # main
    for target_sent in target_texts:
        target_sent_tokens = target_sent.split()
        source_sents_tokens = [doc.split() for doc in source_texts]
        bm25 = BM25Okapi(source_sents_tokens)
        source_sent_scores = list(bm25.get_scores(target_sent_tokens))
        rel_matrix[target_sent] = dict()
        for i, source_sent in enumerate(source_texts):
            rel_matrix[target_sent][source_sent] = source_sent_scores[i]
            all_scores += [source_sent_scores[i]]
    # min-max-normalization
    max_score = max(all_scores)
    min_score = min(all_scores)
    target_texts = list(set(target_texts))
    source_texts = list(set(source_texts))
    for target_sent in target_texts:
        for i, source_sent in enumerate(source_texts):
            score = rel_matrix[target_sent][source_sent]
            try:
                min_max_score = (score - min_score) / (max_score - min_score)
            except:
                min_max_score = 0.0
            rel_matrix[target_sent][source_sent] = min_max_score
    return rel_matrix




def build_text_grams_data(text=None, ngrams=list):
    # init 
    texts = []
    text_grams = []
    # build texts
    if isinstance(text, str) is True:
        texts.append(text)
    elif isinstance(text, list) is True:
        texts += text
    # build text_grams
    for sent in texts:
        sub_text_grams = []
        sent = sent.lower()
        words = sent.split()
        for ngram in ngrams:
            gram_num = int(len(words) / ngram) + 1
            for i in range(gram_num):
                subw = words[i*ngram:(i+1)*ngram]
                subw_ = ' '.join(subw)
                if len(subw) >= 2 and subw_ != '':
                    sub_text_grams.append(subw_)
        if len(sub_text_grams) != 0:
            text_grams += sub_text_grams
        else:
            if sent != '':
                text_grams += [sent]
    return text_grams




def spacy_lemmatizer(texts=list, nlp=None):
    # init
    texts_lemmatizer = []
    doc_gen = nlp.pipe(texts, batch_size=32)
    # main
    for doc in doc_gen:
        texts_lemmatizer.append(' '.join([token.lemma_.lower() for token in doc]))
    return texts_lemmatizer



def KeySentNgram_algo_new(query=str, product_title=str, product_documents=list, mode=None, args=None):
    '''
    recent version, it is too slow. due to spacy_lemmatizer
    '''
    # spacy_lemmatizer
    query =  spacy_lemmatizer(texts=[query], nlp=args.nlp)[0]
    product_title =  spacy_lemmatizer(texts=[product_title], nlp=args.nlp)[0]
    product_documents =  spacy_lemmatizer(texts=product_documents, nlp=args.nlp)
    # init fragments_q, fragments_pd
    fragments_q = build_text_grams_data(text=query, ngrams=[2, 3])
    fragments_pd = build_text_grams_data(text=product_documents, ngrams=[10])
    # query - product_title, product_documents (BM25)
    fragments_pt = build_text_grams_data(text=product_title, ngrams=[5])
    target_texts = fragments_q
    source_texts = fragments_pt + fragments_pd
    extracted_sentsA = ExtractSentByBM25(target_texts=target_texts, source_texts=source_texts)
    if mode == 'BM25' and len(product_documents) != 0:
        # product_title - product_documents (BM25)
        fragments_pt = build_text_grams_data(text=product_title, ngrams=[5, 10])
        target_texts = fragments_pt
        source_texts = fragments_pd
        extracted_sentsB = ExtractSentByBM25(target_texts=target_texts, source_texts=source_texts)
    elif mode == 'PageRank' and len(product_documents) != 0:
        document_rel_matrix = build_rel_matrix(target_texts=fragments_pd, source_texts=fragments_pd)
        document_graph = build_digraph(graph=None, edges=None, mode='init')
        node_list = list(document_rel_matrix.keys())
        document_graph = sync_rel_matrix_into_graph(node_list=node_list, rel_matrix=document_rel_matrix, graph=document_graph)
        adjacency = build_adjacency_matrix(document_graph)
        pagerank = PageRank(n_iter=20)
        pagerank_scores = list(pagerank.fit_transform(adjacency))
        node4score = [[node_list[i], pagerank_scores[i]] for i in range(len(node_list))]
        node4score = sorted(node4score, reverse=True, key=lambda x:x[1])[:len(fragments_q)]
        extracted_sentsB = [node for node, score in node4score]
    else:
        extracted_sentsB = []
    # build extracted_sents
    extracted_sents = extracted_sentsA
    for sent in extracted_sentsB:
        if sent not in extracted_sents:
            extracted_sents.append(sent)
    return extracted_sents


















