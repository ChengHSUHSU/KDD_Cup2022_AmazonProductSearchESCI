
'''
the module of upstream model cannnot generate bert-tokenize by it-self
'''



from transformers import AutoTokenizer, Trainer



bert_model_name = 'roberta-base'
save_path = 'roberta_base_up_'



tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
tokenizer.save_pretrained(save_path)

