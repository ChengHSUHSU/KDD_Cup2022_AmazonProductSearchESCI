

import random
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from data_process import data_process, build_dataloader
from model import CrossEncoder
from bert_model import AUTOTransformer
from util import build_query2passage5score
from util import calculate_eval_score
from util import ndcg_score


'''
microsoft/infoxlm-base,
microsoft/infoxlm-large
'''



# parameter setting
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="None")
parser.add_argument("--model_save_path", type=str, default='save_model/xlm-roberta-large', help="None")
parser.add_argument("--contractive_loss", type=bool, default=False, help="None")
parser.add_argument("--target_query_locale", type=list, default=['us'], help="us, es, jp")
parser.add_argument("--train_val_rate", type=float, default=0.8, help="None")
parser.add_argument("--max_query_length", type=int, default=20, help="None")
parser.add_argument("--max_title_length", type=int, default=60, help="None")
parser.add_argument("--max_super_sents_length", type=int, default=None, help="product_brand + product_color_name + product_bullet_point + product_description")
parser.add_argument("--bert_model_name", type=str, default='xlm-roberta-large', help="it allow [save_model/your_model_name] 'xlm-roberta-large'")
parser.add_argument("--num_labels", type=int, default=-1, help="None")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="None")
parser.add_argument("--lr", type=float, default=7e-6, help="None")
parser.add_argument("--epoch_num", type=int, default=1, help="None")
parser.add_argument("--warmup_steps", type=int, default=5000, help="None")
args = parser.parse_args()



# data process
train_data_x, train_data_y , val_data_x, query2train_data, query2val_data, query2test_data, pd2data = data_process(args=args)


# build dataloader for trainig
train_dataloader = build_dataloader(query2train_data=query2train_data, 
                                    query2val_data=query2val_data, 
                                    query2test_data=query2test_data, 
                                    pd2data=pd2data, 
                                    train_data_x=train_data_x, 
                                    train_data_y=train_data_y, 
                                    args=args)

# modeling module
loss_fct = torch.nn.MSELoss()
model = CrossEncoder(args=args)
if args.contractive_loss is False:
    model.fit(train_dataloader=train_dataloader, loss_fct=loss_fct, args=args)
else:
    model.fit_CL(train_dataloader=train_dataloader, loss_fct=loss_fct, args=args)
model.save(args.model_save_path)


# init auto model
auto_model = AutoModelForSequenceClassification.from_pretrained(args.model_save_path).to(args.device)
auto_trf = AUTOTransformer(bert_model_name=args.bert_model_name, device=args.device)




# evaluation (train data)
ndcg_avg_score_train = []
train_query = list(set([element[0] for element in train_data_x]))
train_query_sample = random.sample(train_query, min(len(val_data_x), len(train_query)))
with torch.no_grad():
    query2passage_pd5score = build_query2passage5score(query_list=train_query_sample, 
                                                       query2data=query2train_data,
                                                       pd2data=pd2data,
                                                       auto_model=auto_model, 
                                                       auto_trf=auto_trf,
                                                       args=args)

    for train_query in tqdm(train_query_sample):
        passage_pd5score = query2passage_pd5score[train_query]
        passage_pd4score = passage_pd5score['mapping_score'][:]
        y_true, y_score = calculate_eval_score(passage_pd4score=passage_pd4score, 
                                               query_data=query2train_data[train_query])


        ndcg_score_train = ndcg_score(y_true=y_true, 
                                      y_score=y_score, 
                                      k=len(y_score), 
                                      gains="exponential")
        
        ndcg_avg_score_train.append(ndcg_score_train)

    count = len(ndcg_avg_score_train)
    ndcg_avg = sum(ndcg_avg_score_train) / len(ndcg_avg_score_train)
    ndcg_std = ((sum([(sc - ndcg_avg) ** 2 for sc in ndcg_avg_score_train])) ** (1/2)) / count
    print('count (train) : ', count)
    print('ndcg_avg (train) : ', ndcg_avg)
    print('ndcg_std (train)  : ', ndcg_std)


# evaluation (validation data)
ndcg_avg_score_val = []
with torch.no_grad():
    query2passage_pd5score = build_query2passage5score(query_list=val_data_x, 
                                                       query2data=query2val_data,
                                                       pd2data=pd2data,
                                                       auto_model=auto_model, 
                                                       auto_trf=auto_trf,
                                                       args=args)
    
    for val_query in tqdm(val_data_x):
        passage_pd5score = query2passage_pd5score[val_query]
        passage_pd4score = passage_pd5score['mapping_score'][:]
        y_true, y_score = calculate_eval_score(passage_pd4score=passage_pd4score, 
                                               query_data=query2val_data[val_query])


        ndcg_score_val = ndcg_score(y_true=y_true, 
                                    y_score=y_score, 
                                    k=len(y_score), 
                                    gains="exponential")
        
        ndcg_avg_score_val.append(ndcg_score_val)

    count = len(ndcg_avg_score_val)
    ndcg_avg = sum(ndcg_avg_score_val) / len(ndcg_avg_score_val)
    ndcg_std = ((sum([(sc - ndcg_avg) ** 2 for sc in ndcg_avg_score_val])) ** (1/2)) / count
    print('count (val) : ', count)
    print('ndcg_avg (val) : ', ndcg_avg)
    print('ndcg_std (val)  : ', ndcg_std)



