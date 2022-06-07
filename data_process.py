


import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses

from util import convert_q_pdi_to_q_sent_feature

 


def data_process(args=None):
    # load data
    print('load data ...')
    path = 'task_1_query-product_ranking/'
    train_path = path + 'train-v0.2.csv'
    test_path = path + 'test_public-v0.2.csv'
    product_path = path + 'product_catalogue-v0.2.csv'
    submit_path = path + 'sample_submission-v0.2.csv'

    train_dat = pd.read_csv(train_path)
    test_dat = pd.read_csv(test_path)
    product_dat = pd.read_csv(product_path)
    submit_dat = pd.read_csv(submit_path)

    # imputattion
    print('imputattion ...')
    train_dat = train_dat.fillna('Empty')
    test_dat = test_dat.fillna('Empty')
    product_dat = product_dat.fillna('Empty')

    # add product_new_id
    print('add product_new_id ...')
    product_dat = build_product_idx(product_dat, locale_name='product_locale')
    train_dat = build_product_idx(train_dat, locale_name='query_locale')
    test_dat = build_product_idx(test_dat, locale_name='query_locale')

    # choose given locale
    print('choose given locale ...')
    dat_lc = train_dat[train_dat['query_locale'].isin(args.target_query_locale)]
    test_dat_lc = test_dat[test_dat['query_locale'].isin(args.target_query_locale)]

    # split train, val data by random_select
    print('split train, val data by random_select ...')
    query_list = list(set(dat_lc['query']))
    train_N = int(args.train_val_rate * len(query_list))
    given_query_list = random.sample(query_list, len(query_list))
    train_query_list = given_query_list[:train_N]
    val_query_list = given_query_list[train_N:]
    train_dat_lc = dat_lc[dat_lc['query'].isin(train_query_list)]
    val_dat_lc = dat_lc[dat_lc['query'].isin(val_query_list)]
    train_pd_set = set(train_dat_lc['product_new_id'])
    val_pd_set = set(val_dat_lc['product_new_id'])
    test_pd_set = set(test_dat_lc['product_new_id'])
    all_pd_list = list(train_pd_set | val_pd_set | test_pd_set)
    given_product_dat = product_dat[product_dat['product_new_id'].isin(all_pd_list)]
    
    # build pd2data
    pd2data = dict()
    for records in tqdm(given_product_dat.to_dict('records')):
        product_id = records['product_id']
        product_new_id = records['product_new_id']
        product_locale = records['product_locale']
        product_title = records['product_title']
        product_bullet_point = records['product_bullet_point']
        product_brand = records['product_brand']
        product_color_name = records['product_color_name']
        product_description = records['product_description']
        origin_super_sents = product_bullet_point.split('\n')
        super_sents = product_brand + '. ' + product_color_name + '. ' + product_bullet_point + '. ' + product_description + '.'
        if product_new_id not in pd2data:
            pd2data[product_new_id] = {
                                    'product_title' : product_title,
                                    'product_bullet_point' : product_bullet_point,
                                    'super_sents' : super_sents,
                                    'origin_super_sents' : origin_super_sents,
                                    'product_brand' : product_brand,
                                    'product_color_name' : product_color_name,
                                    'product_id' : product_id
                                    }
    # build query2data
    print('build query2data ...')
    query2train_data = build_query2data(target_dat=train_dat_lc, target_query_locale=args.target_query_locale)
    query2val_data = build_query2data(target_dat=val_dat_lc, target_query_locale=args.target_query_locale)
    query2test_data = build_query2data(target_dat=test_dat_lc, target_query_locale=args.target_query_locale)


    # build train_data_x, train_data_y
    print('build train_data_x, train_data_y ...')
    train_data_x, train_data_y = [], []
    for query in tqdm(list(query2train_data.keys())):
        pos_set = query2train_data[query]['pos']
        neg_set = query2train_data[query]['neg']
        data_list = query2train_data[query]['data']

        all_pos_set = set([data['product_new_id'] for data in data_list if data['gain'] != 0.0])
        all_neg_set = set(query2train_data[query]['all']) - all_pos_set
        pos_sample = set(random.sample(list(all_pos_set ), min(len(all_pos_set), len(all_neg_set))))

        exact_data = []
        substitute_data = []
        complement_data = []
        irrelevant_data = []
        train_data_x_batch = []
        train_data_y_batch = []
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
            if args.contractive_loss is True:
                index = 0
                while index != len(train_data_x_batch):
                    neg_pdi = neg_data[index % len(neg_data)]
                    pos_pdi = random.sample(pos_data, 1)[0]
                    train_data_x_batch[index] = train_data_x_batch[index] + [pos_pdi, neg_pdi]
                    index +=1
            train_data_x += train_data_x_batch
            train_data_y += train_data_y_batch
    
    # build val_data_x
    print('build val_data_x ...')
    val_data_x = []
    for query in tqdm(list(query2val_data.keys())):
        val_data_x.append(query)

    return train_data_x, train_data_y , val_data_x, query2train_data, query2val_data, query2test_data, pd2data





def build_product_idx(dat, locale_name='product_locale'):
    pd_idx_list = []
    for records in tqdm(dat.to_dict('records')):
        product_id = records['product_id']
        product_locale = records[locale_name]
        pd_idx = product_id + '@' + product_locale
        pd_idx_list.append(pd_idx)
    dat['product_new_id'] = pd_idx_list
    return dat





def build_query2data(target_dat, target_query_locale):
    esci_label2gain = {
                       'exact' : 1,
                       'substitute' : 0.1,
                       'complement' : 0.01,
                       'irrelevant' : 0.0,
                      }
    query2data = dict()
    for records in tqdm(target_dat.to_dict('records')):
        query = records['query']
        query_id = records['query_id']
        product_new_id = records['product_new_id']
        query_locale = records['query_locale']
        product_id = records['product_id']
        product_locale = product_new_id.split('@')[1]
        if query_locale in target_query_locale and query not in query2data:
            query2data[query] = {
                                 'pos' : [],
                                 'neg' : [],
                                 'all' : [],
                                 'locale' : query_locale,
                                 'query_id' : query_id,
                                 'data' : []
                                 }
        if 'esci_label' in records:
            if records['esci_label'] == 'exact':
                query2data[query]['pos'].append(product_new_id)
            else:
                query2data[query]['neg'].append(product_new_id)
            gain = esci_label2gain[records['esci_label'] ]
        else:
            gain = None
        query2data[query]['all'].append(product_new_id)
        query2data[query]['data'].append({
                                          'gain' : gain, 
                                           'product_new_id' : product_new_id, 
                                           'product_id':product_id
                                         })
    return query2data




 
def build_dataloader(query2train_data=dict, 
                     query2val_data=dict, 
                     query2test_data=dict, 
                     pd2data=dict, 
                     train_data_x=None, 
                     train_data_y=None, 
                     args=None):

    # convert query_id, pdi into text
    head_tail_list, sent_length = convert_q_pdi_to_q_sent_feature(q_pdi_list=train_data_x,
                                                                  pd2data=pd2data,
                                                                  eval_mode=False,
                                                                  args=args)
    label_list = train_data_y
    
    # convert into train_dataloader
    train_samples = []
    if args.contractive_loss is False:
        for i, (query, passage) in enumerate(head_tail_list):
            gain_y = label_list[i]
            train_samples.append(InputExample(texts=[query, passage], label=float(gain_y)))
    else:
        for i, (query, passage, pos, neg) in enumerate(head_tail_list):
            gain_y = label_list[i]
            train_samples.append(InputExample(texts=[query, passage, pos, neg], label=float(gain_y)))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    return train_dataloader



