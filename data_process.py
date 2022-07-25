   

import re
import random
import pickle
import pandas as pd
from tqdm import tqdm

from utils import convert_q_pdi_to_q_sent_feature
from utils import convert_pdi_pdi_to_sent_feature

from torch.utils.data import DataLoader
from sentence_transformers import InputExample




def data_info_process(args=None):
    # load data from pickle
    if args.model_cfg['downstream_load_pkl'] is True:
        try:
            data_info_path = args.model_cfg['data_info_path']
            target_fold = args.model_cfg['target_fold']
            path = data_info_path + '_{}fold.pkl'.format(str(target_fold))
            with open(path, "rb") as f:
                data = pickle.load(f)
            data_info = data['data_info']
            return data_info
        except:
            print('[Warning] : Cannot load train-data.pkl..., please check config again')

    # load data (task1, task2)
    print('load data ...')
    task1_path = args.data_process_cfg['task1_path']
    train_path = task1_path + 'train-v0.2.csv'
    test_path = task1_path + 'test_public-v0.3.csv'
    product_path = task1_path + 'product_catalogue-v0.2.csv'
    submit_path = task1_path + 'sample_submission-v0.2.csv'
    train_dat = pd.read_csv(train_path)
    test_dat = pd.read_csv(test_path)
    product_dat = pd.read_csv(product_path)
    submit_dat = pd.read_csv(submit_path)
    #task2_path = args.data_process_cfg['task2_path']
    #task2_train_path = task2_path + 'train-v0.2.csv'
    #task2_product_path = task2_path + 'product_catalogue-v0.2.csv'
    #task2_train_dat = pd.read_csv(task2_train_path)

    # imputation
    print('imputation ...')
    train_dat = train_dat.fillna('Empty')
    test_dat = test_dat.fillna('Empty')
    product_dat = product_dat.fillna('Empty')
    #task2_train_dat = task2_train_dat.fillna('Empty')

    # add product_new_id
    print('add product_new_id ...')
    product_dat = build_product_idx(product_dat, locale_name='product_locale')
    train_dat = build_product_idx(train_dat, locale_name='query_locale')
    test_dat = build_product_idx(test_dat, locale_name='query_locale')
    #task2_train_dat = build_product_idx(task2_train_dat, locale_name='query_locale')

    # build pd2data
    print('build pd2data ...')
    pd2data = build_pd2data(given_product_dat=product_dat)

    # choose given locale
    print('choose given locale ...')
    target_query_locale = args.data_process_cfg['target_query_locale']
    dat_lc = train_dat[train_dat['query_locale'].isin(target_query_locale)]
    test_dat_lc = test_dat[test_dat['query_locale'].isin(target_query_locale)]
    #task2_train_dat_lc = task2_train_dat[task2_train_dat['query_locale'].isin(target_query_locale)]

    # split train, val data by random_select (task1)
    print('split train, val data by random_select (N-fold)...')
    # train_dat_lc, val_dat_lc = split_train_val_data(dat_lc=dat_lc, 
    #                                                 task2_dat_lc=None, 
    #                                                 pd2data=pd2data,
    #                                                 args=args)
    data_lc_Nfolds = split_train_val_data_by_Nfold(dat_lc=dat_lc, args=args)

    for fold, (train_dat_lc, val_dat_lc) in enumerate(data_lc_Nfolds):
        # build task2_complement_dat_lc
        # task2_complement_dat_lc = build_task2_complement_dat(train_dat_lc=train_dat_lc, 
        #                                                      val_dat_lc=val_dat_lc, 
        #                                                      task2_train_dat_lc=task2_train_dat_lc,
        #                                                      pd2data=pd2data,
        #                                                      args=args)

        # build query2data
        print('build query2data ...')
        target_query_locale = args.data_process_cfg['target_query_locale']
        query2train_data = build_query2data(target_dat=train_dat_lc, target_query_locale=target_query_locale, args=args)
        query2val_data = build_query2data(target_dat=val_dat_lc, target_query_locale=target_query_locale, args=args)
        query2test_data = build_query2data(target_dat=test_dat_lc, target_query_locale=target_query_locale, args=args)
        #query2complement_data = build_query2data(target_dat=task2_complement_dat_lc, target_query_locale=target_query_locale)

        # build train_data_x, train_data_y 
        print('build train_data_x, train_data_y ...')
        train_data_x, train_data_y = [], []
        train_data_x, train_data_y = update_train_data_x_y(query2data=query2train_data, 
                                                           train_data_x=train_data_x, 
                                                           train_data_y=train_data_y, 
                                                           pd2data=pd2data,
                                                           args=args,
                                                           train_mode='regression')
        # build train_data_x, train_data_y 
        print('build train_data_x, train_data_y ...')
        val_data_x, val_data_y = [], []
        val_data_x, val_data_y = update_train_data_x_y(query2data=query2val_data, 
                                                       train_data_x=val_data_x, 
                                                       train_data_y=val_data_y, 
                                                       pd2data=pd2data,
                                                       args=args,
                                                       train_mode='regression')
        # build val_data_test
        print('build val_data_test ...')
        val_data_test = list(query2val_data.keys())

        # output
        data_info = {
                     'train_data_x' : train_data_x,
                     'train_data_y' : train_data_y,
                     'query2train_data' : query2train_data,
                     'query2val_data' : query2val_data,
                     'query2test_data' : query2test_data,
                     'pd2data' : pd2data,
                     'val_data_test' : val_data_test
                    }
        
        # save data_info
        data_info_path = args.data_process_cfg['data_info_path']
        data_info_Nfold_path = data_info_path + '_{}fold.pkl'.format(str(fold))
        data = {'data_info' : data_info}
        with open(data_info_Nfold_path, "wb") as f:
            pickle.dump(data, f)
    print('''
          [Warning] : Dont move on modeling_stage, because the stage is for data_process.
          If expect to move on modeling_stage, please set downstream_load_pkl = True.
          ''')
    quit()




def build_product_idx(dat, locale_name='product_locale'):
    pd_idx_list = []
    for records in dat.to_dict('records'):
        product_id = records['product_id']
        product_locale = records[locale_name]
        pd_idx = product_id + '@' + product_locale
        pd_idx_list.append(pd_idx)
    dat['product_new_id'] = pd_idx_list
    return dat




def build_pd2data(given_product_dat=None):
    pd2data = dict()
    for records in given_product_dat.to_dict('records'):
        product_id = records['product_id']
        product_new_id = records['product_new_id']
        product_locale = records['product_locale']
        product_title = records['product_title']
        product_bullet_point = records['product_bullet_point']
        product_brand = records['product_brand']
        product_color_name = records['product_color_name']
        product_description = records['product_description']
        product_description = cleanhtml(raw_html=product_description)
        origin_super_sents = product_bullet_point.split('\n')
        super_sents = product_brand + '. ' \
                    + product_color_name + '. ' \
                    + product_bullet_point + '. ' \
                    + product_description + '.'
        if product_new_id not in pd2data:
            pd2data[product_new_id] = {
                                       'product_title' : product_title,
                                       'product_bullet_point' : product_bullet_point,
                                       'super_sents' : super_sents,
                                       'origin_super_sents' : origin_super_sents,
                                       'product_brand' : product_brand,
                                       'product_color_name' : product_color_name,
                                       'product_id' : product_id,
                                       'product_locale' : product_locale,
                                       'product_description' : product_description
                                      }
    return pd2data




def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>') 
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext




def split_train_val_data(dat_lc=None, task2_dat_lc=None, pd2data=dict, args=None):
    # init parameter
    train_val_rate = args.data_process_cfg['train_val_rate']
    val_data_source = args.data_process_cfg['val_data_source']
    # main
    query_list = list(set(dat_lc['query']))
    if val_data_source == 'task1':
        train_N = int(args.train_val_rate * len(query_list))
        given_query_list = random.sample(query_list, len(query_list))
        train_query_list = given_query_list[:train_N]
        val_query_list = given_query_list[train_N:]
    elif val_data_source == 'task2':
        val_N = int((1-train_val_rate) * len(query_list))
        task2_query = take_legal_query_from_task2(dat_lc=dat_lc, 
                                                  task2_dat_lc=task2_dat_lc, 
                                                  pd2data=pd2data,
                                                  pdi_occupy_rate=1.0,
                                                  args=args)
        task2_query_list = list(set(task2_dat_lc['query']) & set(task2_query))
        given_query_list = random.sample(task2_query_list, len(task2_query_list))
        train_query_list = query_list
        val_query_list = given_query_list[:val_N]
    # take train_dat_lc, val_dat_lc base on query, products
    train_dat_lc = dat_lc[dat_lc['query'].isin(train_query_list)]
    if val_data_source == 'task1': 
        val_dat_lc = dat_lc[dat_lc['query'].isin(val_query_list)]
    elif val_data_source == 'task2': 
        task1_pd_set = set(pd2data.keys())
        val_dat_lc = task2_dat_lc[task2_dat_lc['query'].isin(val_query_list)]
        val_dat_lc = val_dat_lc[val_dat_lc['product_new_id'].isin(task1_pd_set)]
    return train_dat_lc, val_dat_lc



def split_train_val_data_by_Nfold(dat_lc=None, args=None):
    # init
    data_lc_Nfolds = []

    # init parameter
    Nfold = args.data_process_cfg['Nfold']

    # init Nfold_idxs
    n_sample = len(set(dat_lc['query']))
    base_val = int(n_sample / Nfold)
    Nfold_idxs = [0] + [i for i in range(1, n_sample) if i % base_val == 0]
    if n_sample - 1 not in Nfold_idxs:
        Nfold_idxs = Nfold_idxs[:-1]
        Nfold_idxs.append(n_sample- 1)

    # random query_list
    query_list = list(set(dat_lc['query']))
    query_list = random.sample(query_list, len(query_list))

    # main
    for i in range(len(Nfold_idxs)-1):
        st_idx, ed_idx = Nfold_idxs[i], Nfold_idxs[i+1]
        # take train, val query
        val_query_list = query_list[st_idx : ed_idx + 1]
        train_query_list = query_list[ : st_idx] + query_list[ed_idx + 1 : ]
        # take train, val data
        train_dat_lc = dat_lc[dat_lc['query'].isin(train_query_list)]
        val_dat_lc = dat_lc[dat_lc['query'].isin(val_query_list)]
        data_lc_Nfolds.append([train_dat_lc, val_dat_lc])
    return data_lc_Nfolds






def take_legal_query_from_task2(dat_lc=None, task2_dat_lc=None, pd2data=dict, pdi_occupy_rate=float, args=None):
    # init parameter
    target_query_locale = args.data_process_cfg['target_query_locale']

    # init task2_query
    task2_query = [] 
    
    # build task1_query_set
    task1_query_set = set(dat_lc['query'])
    
    # build task1_pd_set
    task1_pd_set = set(pd2data.keys())
    
    # build query2task2_data
    query2task2_data = build_query2data(target_dat=task2_dat_lc, target_query_locale=target_query_locale, args=args)

    # build task2_query
    for query in list(query2task2_data.keys()):
        task2_pd_set = set(query2task2_data[query]['all'])
        rate = len(task2_pd_set & task1_pd_set) / len(task2_pd_set)
        if rate >= pdi_occupy_rate and query not in task1_query_set:
            task2_query.append(query)
    return task2_query




def build_query2data(target_dat=None, target_query_locale=list, args=None):
    # init parameter
    updated_classifier_label = args.model_cfg['updated_classifier_label']
    esci_label2gain = {
                       'exact' : 1,
                       'substitute' : 0.1,
                       'complement' : 0.01,
                       'irrelevant' : 0.0,
                       }
    esci_label2class = {
                       'exact' : updated_classifier_label['E'],
                       'substitute' : updated_classifier_label['S'],
                       'complement' : updated_classifier_label['C'],
                       'irrelevant' : updated_classifier_label['I'],
                       }

    query2data = dict()
    for records in target_dat.to_dict('records'):
        query = records['query']
        product_new_id = records['product_new_id']
        query_locale = records['query_locale']
        product_id = records['product_id']
        if 'query_id'  in records:
            query_id = records['query_id']
        else:
            query_id = None
        product_locale = product_new_id.split('@')[1]
        if query_locale in target_query_locale and query not in query2data:
            query2data[query] = {
                                 'pos' : [],
                                 'neg' : [],
                                 'all' : [],
                                 'locale' : query_locale,
                                 'query_id' : query_id,
                                 'data' : [],
                                 'data_class' : []
                                 } 
        if 'esci_label' in records:
            if records['esci_label'] == 'exact':
                query2data[query]['pos'].append(product_new_id)
            else:
                query2data[query]['neg'].append(product_new_id)
            gain = esci_label2gain[records['esci_label'] ]
            class_ = esci_label2class[records['esci_label']]
        else:
            gain = None
            class_ = None
        query2data[query]['all'].append(product_new_id)
        query2data[query]['data'].append({
                                          'gain' : gain, 
                                           'product_new_id' : product_new_id, 
                                           'product_id':product_id
                                         })
        query2data[query]['data_class'].append({
                                          'gain' : class_, 
                                           'product_new_id' : product_new_id, 
                                           'product_id':product_id
                                         })
    return query2data




def build_task2_complement_dat(train_dat_lc=None, val_dat_lc=None, task2_train_dat_lc=None, pd2data=dict, args=None):
    # build task2_query
    task2_query = take_legal_query_from_task2(dat_lc=train_dat_lc, 
                                              task2_dat_lc=task2_train_dat_lc, 
                                              pd2data=pd2data,
                                              pdi_occupy_rate=0.6,
                                              args=args)
    
    # remove val_query by query_id
    val_query_set = set(val_dat_lc['query'])
    task2_query_wo_val = set([query for query in task2_query if query not in val_query_set])
    
    # build task2_complement_dat
    task2_complement_dat = task2_train_dat_lc[task2_train_dat_lc['query'].isin(task2_query_wo_val)]

    # remove illigual product_id
    old_pdi_set = set(train_dat_lc['product_new_id'])
    task1_pdi_set = set(pd2data.keys())
    update_task1_pdi_set = task1_pdi_set
    task2_complement_dat = task2_complement_dat[task2_complement_dat['product_new_id'].isin(task1_pdi_set)]
    return task2_complement_dat




def update_train_data_x_y(query2data=dict, train_data_x=list, train_data_y=list, pd2data=dict, train_mode=str, args=None):
    # init
    complement_num = 0
    if train_mode == 'regression':
        data_mode = 'data'
    elif train_mode == 'classifier':
        data_mode = 'data' 
    # main
    for query in tqdm(list(query2data.keys())):
        pos_set = query2data[query]['pos']
        neg_set = query2data[query]['neg']
        data_list = query2data[query][data_mode]

        # init container
        lower_bound = None
        upper_bound = None
        pos_pdi_gain_list = [[data['product_new_id'], data['gain']] for data in data_list if data['gain'] == 1.0]
        neg_pdi_gain_list = [[data['product_new_id'], data['gain']] for data in data_list if data['gain'] != 1.0]

        # calculate positive_ratio
        pos_num = len(pos_pdi_gain_list)
        neg_num = len(neg_pdi_gain_list)
        positive_ratio = pos_num / (pos_num + neg_num)
        
        # determine what number of complement; N
        pos_complement_num, neg_complement_num = 0, 0
        if lower_bound is not None and positive_ratio <= lower_bound:
            pos_complement_num = neg_num - pos_num
        if upper_bound is not None and positive_ratio >= upper_bound:
            neg_complement_num = pos_num - neg_num

        pos_complement_data, neg_complement_data = [], []
        
        # update complement_num
        complement_num += len(pos_complement_data + neg_complement_data)

        exact_data = []
        substitute_data = []
        complement_data = []
        irrelevant_data = [] 
        train_data_x_batch = []
        train_data_y_batch = []
        # originnal data (complement_data have no any association with pos/neg complement_data)
        for data in data_list:
            product_new_id = data['product_new_id']
            gain = data['gain']
            train_data_x_batch.append([query, product_new_id])
            train_data_y_batch.append(gain)
            if gain == 1.0:
                exact_data.append(product_new_id)
            elif gain == 0.1:
                substitute_data.append(product_new_id)
            elif gain == 0.01:
                complement_data.append(product_new_id)
            else:
                irrelevant_data.append(product_new_id)
        # pseudo data
        for product_new_id, gain in pos_complement_data + neg_complement_data:
            train_data_x_batch.append([query, product_new_id])
            train_data_y_batch.append(gain)
   

        if len(train_data_x_batch) > 0:
            if len(complement_data + irrelevant_data) > 0:
                if len(exact_data) + len(substitute_data) > 0:
                    neg_data = complement_data + irrelevant_data
                    pos_data = exact_data + substitute_data
                else:
                    if len(complement_data) > 0 and len(irrelevant_data) > 0:
                        neg_data = irrelevant_data
                        pos_data = complement_data
                    else:
                        neg_data = irrelevant_data + complement_data
                        pos_data = irrelevant_data + complement_data
            elif len(complement_data + irrelevant_data) == 0:
                if len(exact_data) > 0 and len(substitute_data) > 0:
                    neg_data = substitute_data
                    pos_data = exact_data
                else:
                    neg_data = substitute_data + exact_data
                    pos_data = substitute_data + exact_data
            train_data_x += train_data_x_batch
            train_data_y += train_data_y_batch
    # convert gain to label_name (hard code)
    for i in range(len(train_data_y)):
        if train_data_y[i] == 1.0:
            train_data_y[i] = 'E'
        elif train_data_y[i] == 0.1:
            train_data_y[i] = 'S'
        elif train_data_y[i] == 0.01:
            train_data_y[i] = 'C'
        else:
            train_data_y[i] = 'I'
    return train_data_x, train_data_y




def build_dataloader(train_data_x=None, 
                     train_data_y=None,
                     pd2data=dict,
                     args=None,
                     drop_last=True):
    # init parameter
    batch_size = args.model_cfg['batch_size']
    use_margin_rank_loss = args.model_cfg['use_margin_rank_loss']
    

    if use_margin_rank_loss is False:
        head_tail_list, sent_length = convert_q_pdi_to_q_sent_feature(q_pdi_list=train_data_x,
                                                                      pd2data=pd2data,
                                                                      eval_mode=False,
                                                                      args=args)
    else:
        head_tail_list, sent_length = convert_pdi_pdi_to_sent_feature(q_pdi_pdi_list=train_data_x, 
                                                                      pd2data=pd2data, 
                                                                      args=args)
    if train_data_y is None:
        label_list = [0.0 for _ in range(len(train_data_x))]
    else:
        label_list = train_data_y
    
    # convert into train_dataloader 
    train_samples = []
    if use_margin_rank_loss is False:
        for i, (query, passage) in enumerate(head_tail_list):
            gain_y = label_list[i]
            train_samples.append(InputExample(texts=[query, passage], label=gain_y))
    else:
        for i, (query, left_passage, right_passage) in enumerate(head_tail_list):
            gain_y = label_list[i]
            train_samples.append(InputExample(texts=[query, left_passage, right_passage], label=gain_y))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size, drop_last=drop_last)
    return train_dataloader


 


def additional_data_process(data_info=dict, args=None):
    '''
    1. two_stage_replace_label (X)
    2. margin rank data
    '''
    # init parameter
    #updated_regression_label = args.model_cfg['updated_regression_label']
    use_margin_rank_loss = args.model_cfg['use_margin_rank_loss']
    
    # train data
    train_data_x = data_info['train_data_x']
    train_data_y = data_info['train_data_y']
    train_data = [train_data_x, train_data_y]
    
    # replace label for two_stage
    # if updated_regression_label is not None:
    #     # updayed label
    #     E_label = updated_regression_label[0]
    #     S_label = updated_regression_label[1]
    #     C_label = updated_regression_label[2]
    #     I_label = updated_regression_label[3]
    #     # update
    #     train_data_x_update = []
    #     train_data_y_update = []
    #     for i, gain in enumerate(train_data_y):
    #         x = train_data_x[i]
    #         if gain == 1:
    #             train_data_x_update.append(x)
    #             train_data_y_update.append(E_label)
    #         elif gain == 0.1:
    #             train_data_x_update.append(x)
    #             train_data_y_update.append(S_label)
    #         elif gain == 0.01:
    #             train_data_x_update.append(x)
    #             train_data_y_update.append(C_label)
    #         elif gain == 0.0:
    #             train_data_x_update.append(x)
    #             train_data_y_update.append(I_label)
    #     train_data_x = train_data_x_update
    #     train_data_y = train_data_y_update

    # margin rank data
    if use_margin_rank_loss is True:
        query2train_data = data_info['query2train_data']
        query_locale2query2score = data_info['query_locale2query2score']
        train_data_x, train_data_y = build_margin_rank_data(query2train_data=query2train_data, 
                                                            query_locale2query2score=query_locale2query2score, 
                                                            train_data=train_data, 
                                                            args=args)
    data_info['train_data_x'] = train_data_x
    data_info['train_data_y'] = train_data_y
    return data_info





def build_margin_rank_data(query2train_data=dict, query_locale2query2score=None, train_data=list, args=None):
    '''
    1. |sample_E_S|, |sample_E_CI|, |sample_S_CI| > 0
    2. minmax(|E_S|, |E_CI|, |S_CI|)  = P_E_S, P_E_CI, P_S_CI
    3. random_choice(E_S, E_CI, S_CI, weight), where weight = (P_E_S, P_E_CI, P_S_CI)
    '''
    # init parameter 
    margin_rank_sample_rate = args.model_cfg['margin_rank_sample_rate']
    # init train_data_x, train_data_y
    train_data_x, train_data_y = [], []
    # build query_list
    query_list_update, weights = [], []
    query_list = list(query2train_data.keys())
    sample_n = int(margin_rank_sample_rate * len(query_list))
    idx_list = [i for i in range(len(query_list))]
    use_train_ndcg_as_sample_prod = True
    if use_train_ndcg_as_sample_prod is True:
        for i, query in tqdm(enumerate(query_list)):
            query_locale = query2train_data[query]['locale']
            if query in query_locale2query2score[query_locale]:
                score = query_locale2query2score[query_locale][query]
            else:
                score = 0.1
                print('[Warning] : if query not in query_locale2query2score[query_locale]..')
            weights.append(score)
        sum_ = sum(weights)
        weights = [val/sum_ for val in weights]
        sample_idxs = np.random.choice(idx_list, sample_n, p=weights, replace=False)
        for i in sample_idxs:
            query_list_update.append(query_list[i])
        query_list = query_list_update
    # main
    #query_list = query_list[:10]
    for query in tqdm(query_list):
        # init train_data_x_sm, train_data_y_sm
        train_data_x_sm = []
        train_data_y_sm = []
        #query_locale = query2train_data['locale']
        # data_list
        data_list = query2train_data[query]['data']
        # determine sample_n
        sample_n = len(data_list) * 2
        # build label2pdi_list
        label2pdi_list = {'E' : [], 'S' : [], 'CI' : []}
        for record in data_list:
            gain = record['gain']
            product_new_id = record['product_new_id']
            if gain == 1.0:
                label2pdi_list['E'].append(product_new_id)
            elif gain == 0.1:
                label2pdi_list['S'].append(product_new_id)
            else:
                label2pdi_list['CI'].append(product_new_id)
        # build pair_label2pair_pdi_list
        pair_label2pair_pdi_list = {'E-S' : [], 'E-CI' : [], 'S-CI' : []}
        for label_t in ['E', 'S', 'CI']:
            pdi_list_t = label2pdi_list[label_t]
            for label_s in ['E', 'S', 'CI']:
                pdi_list_s = label2pdi_list[label_s]
                pair_label = label_t + '-' + label_s
                if pair_label in pair_label2pair_pdi_list:  
                    for pdi_t in pdi_list_t:
                        for pdi_s in pdi_list_s:
                            pair_label2pair_pdi_list[pair_label].append([pdi_t, pdi_s])
        # randomly select one as data
        for pair_label in ['E-S', 'E-CI', 'S-CI']:
            if len(pair_label2pair_pdi_list[pair_label]) > 0:
                pair_pdi = random.sample(pair_label2pair_pdi_list[pair_label], 1)[0]
                train_data_x_sm.append([query] + pair_pdi)
                sample_n -=1
        # min-max normalization
        ES_num = len(pair_label2pair_pdi_list['E-S'])
        ECI_num = len(pair_label2pair_pdi_list['E-CI'])
        SCI_num = len(pair_label2pair_pdi_list['S-CI'])
        sum_ = sum([ES_num, ECI_num, SCI_num])
        if sum_ != 0:
            P_ES = ES_num / sum_
            P_ECI = ECI_num / sum_
            P_SCI = SCI_num / sum_
        else:
            P_ES, P_ECI, P_SCI = 0,0,0
        # determine expectation_value
        pair_label2expect_num = {
                                 'E-S' : int(P_ES * sample_n), 
                                 'E-CI' : int(P_ECI * sample_n),
                                 'S-CI' : int(P_SCI * sample_n)
                                }
        # randomly select data by expect_value
        non_zero_num = 0
        for pair_label in ['E-S', 'E-CI', 'S-CI']:
            pair_pdi_list = pair_label2pair_pdi_list[pair_label]
            expect_num = pair_label2expect_num[pair_label]
            if len(pair_pdi_list) > 0 and expect_num > 0:
                non_zero_num += 1
        if non_zero_num >= 1:

            for pair_label in ['E-S', 'E-CI', 'S-CI']:
                pair_pdi_list = pair_label2pair_pdi_list[pair_label]
                if non_zero_num != 1:
                    expect_num = pair_label2expect_num[pair_label]
                else:
                    expect_num = 5
                if len(pair_pdi_list) > 0 and expect_num > 0:
                    sample_pair_pdi_list = random.sample(pair_pdi_list, min(expect_num, len(pair_pdi_list)))
                    for pair_pdi in sample_pair_pdi_list:
                        train_data_x_sm.append([query] + pair_pdi) 
            # randomly select half data as posm neg data
            idx_list = [i for i in range(len(train_data_x_sm))]
            if len(train_data_x_sm) > 1:
                sample_pos_idx_set = set(random.sample(idx_list, int(len(idx_list) / 2)))
            else:
                sample_pos_idx_set = set(train_data_x_sm)
            for idx in idx_list:
                if idx not in sample_pos_idx_set:
                    # swap
                    train_data_x_sm[idx][2], train_data_x_sm[idx][1] = train_data_x_sm[idx][1], train_data_x_sm[idx][2]
                    train_data_y_sm.append(-1)
                else:
                    train_data_y_sm.append(1)
            # coolect train_data_x, train_data_y
            train_data_x += train_data_x_sm
            train_data_y += train_data_y_sm
    return train_data_x, train_data_y










