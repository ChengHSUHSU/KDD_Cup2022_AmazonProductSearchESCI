





from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from bert_model_refactor import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BertForSequenceClassification
#from transformers import RobertaForSequenceClassification


logger = logging.getLogger(__name__)





class CrossEncoder:
    def __init__(self, args=None, tokenizer_args=dict(), automodel_args=dict()):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.
        It does not yield a sentence embedding and does not work for individually sentences.
        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param automodel_args: Arguments passed to AutoModelForSequenceClassification
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """
        # bert config
        self.config = AutoConfig.from_pretrained(args.bert_model_name)
        
        # bert tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, **tokenizer_args)

        # num_labels
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if args.num_labels == -1 and not classifier_trained:
            num_labels = 1

        self.config.num_labels = num_labels

        # init model
        if 'roberta' in args.bert_model_name:
            self.model = RobertaForSequenceClassification.from_pretrained(args.bert_model_name, config=self.config, **automodel_args)
        else: 
            self.model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_name, config=self.config, **automodel_args)
        
        # max_length
        if args.max_super_sents_length is None:
            max_super_sents_length = 0
        self.max_length = args.max_query_length + args.max_title_length + max_super_sents_length

        # default_activation_function
        self.default_activation_function = torch.nn.Identity()
        try:
            self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
        except Exception as e:
            logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        # elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
        #     self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        # else:
        #     self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()
        
        # device
        self._target_device = torch.device(args.device) 


    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels


    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized


    def smart_batching_collate_for_triple(self, batch):
        texts = [[] for _ in range(2)]
        left_texts = [[] for _ in range(2)]
        right_texts = [[] for _ in range(2)]
        labels = []

        for example in batch:
            # get object
            head = example.texts[0].strip()
            tail = example.texts[1].strip()
            left_tail = example.texts[2].strip()
            right_tail = example.texts[3].strip()
            # left, right
            left_texts[0].append(head)
            right_texts[0].append(head)
            texts[0].append(head)
            left_texts[1].append(left_tail)
            right_texts[1].append(right_tail)
            texts[1].append(tail)
            # label
            labels.append(example.label)

        tokenized = self.tokenizer(*texts, 
                            padding=True, 
                            truncation='longest_first', 
                            return_tensors="pt", 
                            max_length=self.max_length)
        left_tokenized = self.tokenizer(*left_texts, 
                            padding=True, 
                            truncation='longest_first', 
                            return_tensors="pt", 
                            max_length=self.max_length)
        right_tokenized = self.tokenizer(*right_texts, 
                            padding=True, 
                            truncation='longest_first', 
                            return_tensors="pt", 
                            max_length=self.max_length)
        labels = torch.tensor(labels, 
                            dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in left_tokenized:
            left_tokenized[name] = left_tokenized[name].to(self._target_device)
        for name in right_tokenized:
            right_tokenized[name] = right_tokenized[name].to(self._target_device)
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)
        return tokenized, labels, left_tokenized, right_tokenized


    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            args=None):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        optimizer_params = {'lr': args.lr}
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if args.model_save_path is not None:
            os.makedirs(args.model_save_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * args.epoch_num)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=args.warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in trange(args.epoch_num, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, args.model_save_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, args.model_save_path, save_best_model, epoch, -1, callback)


    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.
        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)


    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)


    def fit_CL(self,
               train_dataloader: DataLoader,
               evaluator: SentenceEvaluator = None,
               loss_fct = None,
               activation_fct = nn.Identity(),
               scheduler: str = 'WarmupLinear',
               optimizer_class: Type[Optimizer] = torch.optim.AdamW,
               weight_decay: float = 0.01,
               evaluation_steps: int = 0,
               save_best_model: bool = True,
               max_grad_norm: float = 1,
               use_amp: bool = False,
               callback: Callable[[float, int, int], None] = None,
               show_progress_bar: bool = True,
               args=None):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        optimizer_params = {'lr': args.lr}
        train_dataloader.collate_fn = self.smart_batching_collate_for_triple

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if args.model_save_path is not None:
            os.makedirs(args.model_save_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * args.epoch_num)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=args.warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in trange(args.epoch_num, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels, pos_features, neg_features in tqdm(train_dataloader, 
                                                                     desc="Iteration", 
                                                                     smoothing=0.05, 
                                                                     disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                        loss_CL = self.model.forward_with_CL(pos_bert_inputs=pos_features, 
                                                             neg_bert_inputs=neg_features, 
                                                             margin=0.5, 
                                                             return_dict=True)
                    alpha = 0.1
                    loss_total = (-1) * loss_CL + alpha * loss_value
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_CL = self.model.forward_with_CL(pos_bert_inputs=pos_features, 
                                                         neg_bert_inputs=neg_features, 
                                                         margin=0.5, 
                                                         return_dict=True)
                    alpha = 0.1
                    loss_total = (-1) * loss_CL + alpha * loss_value
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, args.model_save_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, args.model_save_path, save_best_model, epoch, -1, callback)






