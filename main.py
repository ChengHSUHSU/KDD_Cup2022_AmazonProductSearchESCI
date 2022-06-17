
import os
import random
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from data_process import data_process, build_dataloader, data_process_denoise
from model import CrossEncoder, Learning_From_FailurePrediction
from bert_model import AUTOTransformer
from util import build_query2passage5score
from util import calculate_eval_score
from util import ndcg_score
from util import build_submit_result
from util import evaluation


# setting cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# parameter setting (O)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="None")
parser.add_argument("--model_save_path", type=str, default='save_model/roberta_base_dp_t2', help="None")
parser.add_argument("--contractive_loss", type=bool, default=False, help="None")
parser.add_argument("--target_query_locale", type=list, default=['us'], help="us, es, jp")
parser.add_argument("--train_val_rate", type=float, default=0.8, help="None")
parser.add_argument("--max_query_length", type=int, default=20, help="None")
parser.add_argument("--max_title_length", type=int, default=60, help="None")
parser.add_argument("--max_super_sents_length", type=int, default=None, help="product_brand + product_color_name + product_bullet_point + product_description")
parser.add_argument("--bert_model_name", type=str, default='roberta-base', help="it allow [save_model/your_model_name] 'xlm-roberta-large'")
parser.add_argument("--num_labels", type=int, default=-1, help="None")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="None")
parser.add_argument("--lr", type=float, default=7e-6, help="None")
parser.add_argument("--epoch_num", type=int, default=1, help="None")
parser.add_argument("--warmup_steps", type=int, default=5000, help="None")
parser.add_argument("--submit_save_path", type=str, default='save_submit/roberta_base_t2', help="None")

# adversarial learning (X)
parser.add_argument("--al_fgm", type=bool, default=False, help="adversarial_learning_fgm")
parser.add_argument("--al_fgm_epsilon", type=float, default=1.05, help="adversarial_learning_fgm_epsilon")

# use task2 data (?)
parser.add_argument("--use_task2_data", type=bool, default=True, help="None")
parser.add_argument("--task2_used_rate", type=float, default=0.2, help="None")


# learning from failure prediction (X)
parser.add_argument("--failure_threshold", type=float, default=0.80, help="None")
parser.add_argument("--failure_k", type=int, default=1, help="None")

# denoise (X)
parser.add_argument("--denoise_mode", type=str, default=None, help='''
                                                                    None -> dont use it. 
                                                                    random-query, 
                                                                    random-records,
                                                                    undersampling,
                                                                    oversampling
                                                                   ''')
parser.add_argument("--denoise_rate", type=float, default=0.1, help="None")

# upstream (O)
parser.add_argument("--upstream", type=bool, default=False, help="None")
args = parser.parse_args()
print(args)



# data process
train_data_x, train_data_y , val_data_x, query2train_data, query2val_data, query2test_data, pd2data = data_process(args=args)




# data process for denoise
train_data_x, train_data_y = data_process_denoise(train_data_x=train_data_x, 
                                                  train_data_y=train_data_y, 
                                                  query2train_data=query2train_data, 
                                                  args=args) 


# regard exact data as pos_data, otherwise as neg_data
if args.upstream is True:
    train_data_x_update = []
    train_data_y_update = []
    for i, gain in enumerate(train_data_y):
        x = train_data_x[i]
        if gain == 1.0:
            train_data_x_update.append(x)
            train_data_y_update.append(1.0)
        elif gain <= 0.01:
            train_data_x_update.append(x)
            train_data_y_update.append(0.0)
    train_data_x = train_data_x_update
    train_data_y = train_data_y_update




# build dataloader for trainig
train_dataloader = build_dataloader(train_data_x=train_data_x, 
                                    train_data_y=train_data_y,
                                    pd2data=pd2data,
                                    args=args)




# modeling module
if args.upstream is True:
    loss_fct = torch.nn.MSELoss()
else:
    loss_fct = torch.nn.MSELoss()

cross_encoder_model = CrossEncoder(args=args)


if args.failure_k <= 1:
    # only learn by all train data
    if args.contractive_loss is False:
        cross_encoder_model.fit(train_dataloader=train_dataloader, loss_fct=loss_fct, args=args)
    else:
        cross_encoder_model.fit_CL(train_dataloader=train_dataloader, loss_fct=loss_fct, args=args)
else:
    # learning by failure prediction
    cross_encoder_model = Learning_From_FailurePrediction(train_data_x=train_data_x, 
                                                          train_data_y=train_data_y, 
                                                          query2train_data=query2train_data, 
                                                          pd2data=pd2data,
                                                          cross_encoder_model=cross_encoder_model,
                                                          loss_fct=loss_fct,
                                                          args=args)
# save model
cross_encoder_model.save(args.model_save_path)



# init auto model
auto_model = AutoModelForSequenceClassification.from_pretrained(args.model_save_path).to(args.device)
auto_trf = AUTOTransformer(bert_model_name=args.model_save_path, device=args.device)




# evaluation (train data)
train_query = list(set([element[0] for element in train_data_x]))
train_query_sample = random.sample(train_query, min(len(val_data_x), len(train_query)))

failure_pred_query_train = evaluation(query_list=train_query_sample, 
                                      query2data=query2train_data, 
                                      pd2data=pd2data, 
                                      auto_model=auto_model, 
                                      auto_trf=auto_trf, 
                                      args=args,
                                      category='Train')


# evaluation (validation data)
failure_pred_query_val__ = evaluation(query_list=val_data_x, 
                                      query2data=query2val_data, 
                                      pd2data=pd2data, 
                                      auto_model=auto_model, 
                                      auto_trf=auto_trf, 
                                      args=args,
                                      category='Validation')


# inference (submit test data) 
if args.submit_save_path is not None:
    submit_dat = build_submit_result(query2test_data=query2test_data, 
                                     pd2data=pd2data, 
                                     auto_model=auto_model, 
                                     auto_trf=auto_trf, 
                                     args=args)
    submit_dat = pd.DataFrame(submit_dat)
    submit_dat.to_csv('{}.csv'.format(args.submit_save_path), index=False)


  
