

import numpy as np
from tqdm.autonotebook import tqdm, trange

import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from bert_model import AUTOTransformer










class CrossEncoder:
    def __init__(self, args=None):
        # init parameter
        self.args = args
        device = args.model_cfg['device']
        num_labels = args.model_cfg['num_labels']
        bert_model_name = args.model_cfg['bert_model_name']
        max_query_length = args.model_cfg['max_query_length']
        max_title_length = args.model_cfg['max_title_length']
        use_classfier = args.model_cfg['use_classfier']
        use_margin_rank_loss = args.model_cfg['use_margin_rank_loss']
        margin = args.model_cfg['margin']
        classifier_weights = args.model_cfg['classifier_weights']
        self.updated_regression_label = args.model_cfg['updated_regression_label']
        self.updated_classifier_label = args.model_cfg['updated_classifier_label']

        # device
        self._target_device = torch.device(device) 

        # bert config
        self.config = AutoConfig.from_pretrained(bert_model_name)
        
        # bert tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # insert num_labels
        self.config.num_labels = num_labels

        # bert model
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, config=self.config)
        self.model.to(self._target_device)

        # max_length
        self.max_length = max_query_length + max_title_length

        # default_activation_function
        self.default_activation_function = torch.nn.Identity()
        try:
            self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
        except Exception as e:
            logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)))

        # loss_fct
        if use_classfier is True:
            self.loss_fct = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(classifier_weights,
                                                       dtype=torch.float32,
                                                       device=device)) 
        elif use_margin_rank_loss is True:
            self.loss_fct = torch.nn.MarginRankingLoss(margin=margin)
        else:
            self.loss_fct = torch.nn.MSELoss()



    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(self.updated_regression_label[example.label])

        tokenized = self.tokenizer(*texts, 
                                   padding=True, 
                                   truncation='longest_first', 
                                   return_tensors="pt", 
                                   max_length=self.max_length)
        labels = torch.tensor(labels, 
                              dtype=torch.float if self.config.num_labels == 1 \
                              else torch.long).to(self._target_device)
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels



    def smart_batching_collate_classifier(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        onehot = np.eye(self.config.num_labels)

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
            labels.append(self.updated_classifier_label[example.label])

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)
        labels = torch.nn.functional.one_hot(labels, num_classes=self.config.num_labels)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels



    def smart_batching_collate_for_marginrank(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        tokenized_list = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
            labels.append(self.updated_regression_labelp[example.label])
        
        for q_idx, p_idx in [(0, 1), (0, 2)]:
            q_texts = texts[q_idx]
            p_texts = texts[p_idx]
            qp_texts = [q_texts, p_texts]
            
            tokenized = self.tokenizer(*qp_texts, 
                                padding=True, 
                                truncation='longest_first', 
                                return_tensors="pt",
                                max_length=self.max_length)
            for name in tokenized:
                tokenized[name] = tokenized[name].to(self._target_device)
            tokenized_list.append(tokenized)

        labels = torch.tensor(labels, 
                            dtype=torch.float if self.config.num_labels == 1 \
                            else torch.long).to(self._target_device)

        return tokenized_list, labels



    def fit(self,
            train_dataloader=None,
            activation_fct=torch.nn.Identity(),
            scheduler='WarmupLinear',
            optimizer_class=torch.optim.AdamW,
            weight_decay=0.01,
            max_grad_norm=1,
            show_progress_bar=True):
        
        # init parameter
        epoch_num = self.args.model_cfg['epoch_num']
        warmup_steps = self.args.model_cfg['warmup_steps']
        use_classfier = self.args.model_cfg['use_classfier']
        use_margin_rank_loss = self.args.model_cfg['use_margin_rank_loss']

        # init collate_fn
        if use_classfier is True:
            train_dataloader.collate_fn = self.smart_batching_collate_classifier
        elif use_margin_rank_loss is True:
            train_dataloader.collate_fn = self.smart_batching_collate_for_marginrank
        else:
            train_dataloader.collate_fn = self.smart_batching_collate

        # init num_train_steps
        num_train_steps = int(len(train_dataloader) * epoch_num)

        # init optimizers
        optimizer_params = {'lr' : float(self.args.model_cfg['lr'])}
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # setting warm-up scheduler
        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, 
                                                           scheduler=scheduler, 
                                                           warmup_steps=warmup_steps, 
                                                           t_total=num_train_steps)
        # training
        for epoch in trange(epoch_num, desc="Epoch", disable=not show_progress_bar):
            self.model.zero_grad()
            self.model.train()
            for features, labels in tqdm(train_dataloader, 
                                         desc="Iteration", 
                                         smoothing=0.05, 
                                         disable=not show_progress_bar):
                # calculate loss_val
                if use_margin_rank_loss is True:
                    features_left = features[0]
                    features_right = features[1]
                    model_predictions_left = self.model(**features_left, return_dict=True).logits
                    model_predictions_right = self.model(**features_right, return_dict=True).logits
                    loss_value = self.loss_fct(model_predictions_left, model_predictions_right, labels.view(-1,1))
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)

                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                        loss_value = self.loss_fct(logits, labels)
                    else:
                        loss_value = self.loss_fct(logits.view(-1,self.config.num_labels),
                                                   labels.type_as(logits).view(-1,self.config.num_labels))
                # backward to loss_value
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        self.model.eval()



    def save(self, path=str):
        """
        Saves all model and tokenizer to path
        """
        path = self.args.model_cfg['model_save_path']
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)





def load_cross_encoder_model(args=None):
    # init parameter
    device = args.model_cfg['device']
    model_save_path = args.model_cfg['model_save_path']
    # main
    auto_model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(device)
    auto_trf = AUTOTransformer(bert_model_name=model_save_path, device=device)
    return auto_model, auto_trf






