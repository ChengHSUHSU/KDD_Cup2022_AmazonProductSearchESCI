

import torch
import random
import argparse
from transformers import TrainingArguments
from transformers import RobertaForMaskedLM
from transformers import AutoTokenizer, Trainer
from transformers import DataCollatorForWholeWordMask
from transformers import DataCollatorForLanguageModeling

from data_process_refactor import data_process
from util_refactor import build_corpus




# parameter setting
parser = argparse.ArgumentParser()
parser.add_argument("--target_query_locale", type=list, default=['us'], help="us, es, jp")
parser.add_argument("--train_val_rate", type=float, default=0.8, help="None")
parser.add_argument("--bert_model_name", type=str, default='roberta-base', help="it allow [save_model/your_model_name] 'xlm-roberta-large'")
parser.add_argument("--mlm_probability", type=float, default=0.15, help="None")
parser.add_argument("--gradient_accumulation_steps", type=int, default=256, help="None")
parser.add_argument("--lr", type=float, default=5e-5, help="None")
parser.add_argument("--weight_decay", type=int, default=0, help="None")
parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available(), help="None")
parser.add_argument("--batch_size", type=int, default=32, help="None")
parser.add_argument("--epoch_num", type=int, default=2, help="None")
parser.add_argument("--block_size", type=int, default=256, help="None")
parser.add_argument("--model_save_path", type=str, default='save_model/your_model_name', help="None")
args = parser.parse_args()




# init model
model = RobertaForMaskedLM.from_pretrained(args.bert_model_name)
tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)



# data process
train_data_x, train_data_y , val_data_x, query2train_data, query2val_data, query2test_data, pd2data = data_process(args=args)


# build corpus
corpus = build_corpus(pd2data=pd2data)


# sample val
val_corpus = random.sample(corpus, int(len(corpus) * args.train_val_rate))



# build TextDataset
train_dataset = TextDataset(tokenizer=tokenizer, block_size=args.block_size, text_list=corpus)
eval_dataset = TextDataset(tokenizer=tokenizer, block_size=args.block_size, text_list=val_corpus)
data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)


# init training arguments
training_args = TrainingArguments(output_dir='cache', 
                                  do_train=True, 
                                  do_eval=True, 
                                  gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                  learning_rate=args.lr,
                                  weight_decay=args.weight_decay,
                                  no_cuda=not args.cuda,
                                  per_device_train_batch_size=args.batch_size,
                                  per_device_eval_batch_size=args.batch_size,
                                  num_train_epochs=args.epoch_num)

# init trainer
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset)

# modeloing
trainer.train()

# save model
trainer.save_model(args.model_save_path)





