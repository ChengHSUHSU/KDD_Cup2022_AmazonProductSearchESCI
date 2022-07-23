





#
 
import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import AutoTokenizer, BertModel, BertConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel



class RobertaClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        self.distance_metric=SiameseDistanceMetric.COSINE_DISTANCE



    def forward_with_CL(
        self,
        pos_bert_inputs=dict, 
        neg_bert_inputs=dict, 
        margin=0.5,
        return_dict=True):
        '''
        loss = RELU( margin - cos(pos_pool, neg_pool)).pow(2).mean()
        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pos_bert_inputs['return_dict'] = return_dict
        neg_bert_inputs['return_dict'] = return_dict

        pos_outputs = self.roberta(**pos_bert_inputs)
        neg_outputs = self.roberta(**neg_bert_inputs)


        pos_pooled_output = pos_outputs.last_hidden_state[:, 0, :]
        neg_pooled_output = neg_outputs.last_hidden_state[:, 0, :]

        distances = self.distance_metric(pos_pooled_output, neg_pooled_output)
        losses = F.relu(margin - distances).pow(2)
        return losses.mean()



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        self.distance_metric=SiameseDistanceMetric.COSINE_DISTANCE

    def forward_with_CL(
        self,
        pos_bert_inputs=dict, 
        neg_bert_inputs=dict, 
        margin=0.5,
        return_dict=True):
        '''
        loss = RELU( margin - cos(pos_pool, neg_pool)).pow(2).mean()
        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pos_bert_inputs['return_dict'] = return_dict
        neg_bert_inputs['return_dict'] = return_dict

        pos_outputs = self.bert(**pos_bert_inputs)
        neg_outputs = self.bert(**neg_bert_inputs)

        pos_pooled_output = pos_outputs.pooler_output
        neg_pooled_output = neg_outputs.pooler_output

        distances = self.distance_metric(pos_pooled_output, neg_pooled_output)
        losses = F.relu(margin - distances).pow(2)
        return losses.mean()


    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
 

class BertForClassificationContrativeLearning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.distance_metric=SiameseDistanceMetric.COSINE_DISTANCE
    
    def forward_with_CL(
        self,
        pos_bert_inputs=dict, 
        neg_bert_inputs=dict, 
        margin=0.5,
        return_dict=True):
        '''
        loss = RELU( margin - cos(pos_pool, neg_pool)).pow(2).mean()
        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pos_bert_inputs['return_dict'] = return_dict
        neg_bert_inputs['return_dict'] = return_dict

        pos_outputs = self.bert(**pos_bert_inputs)
        neg_outputs = self.bert(**neg_bert_inputs)

        pos_pooled_output = pos_outputs.pooler_output
        neg_pooled_output = neg_outputs.pooler_output

        distances = self.distance_metric(pos_pooled_output, neg_pooled_output)
        losses = F.relu(margin - distances).pow(2)
        return losses.mean()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F





class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)








class AUTOTransformer:
    def __init__(self, bert_model=None, bert_model_name=None, device='cpu'):
        '''
        [DONE]
        '''
        if bert_model_name is None:
            bert_model_name = 'roberta-base'
        self.bert_model = bert_model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.inf = 10 ** 3
        self.device = device
        self.bert_config = BertConfig.from_pretrained(bert_model_name)

 
    def convert_batch_sent_to_bert_input(self, batch_sent=list, 
                                               sent_length=None):
        # init container
        batch_bert_input = dict()
        # main
        
        for sent in batch_sent:
            bert_input = self.convert_sent_to_token(sent=sent, 
                                                    sent_length=sent_length)
            bert_feature_list = list(bert_input.keys())
            for bert_feature in bert_feature_list:
                if 'sent_token' != bert_feature:
                    if bert_feature not in batch_bert_input:
                        batch_bert_input[bert_feature] = []
                    batch_bert_input[bert_feature].append(bert_input[bert_feature])
        
        return batch_bert_input

    
    def convert_sent_to_token(self, sent=None, 
                                    sent_length=None):

        CLS = self.tokenizer.cls_token
        SEP = self.tokenizer.sep_token
        #is_roberta = True if "roberta" in self.bert_config.architectures[0].lower() else False
        is_roberta = False

        # sent to sent_token
        if isinstance(sent, str) is True:
            sent_token = [CLS]
            if sent_length is not None:
                sent_token += self.tokenizer.tokenize(sent)[:sent_length[0]-2]
            else:
                print('[ERROR] : The version cannot allow sent_length as default.')
            sent_token += [SEP]

        elif isinstance(sent, list) is True:
            sent_token = [CLS]
            head_text = sent[0]
            length = sent_length[0]
            text_token = self.tokenizer.tokenize(head_text)[:length-1]
            sent_token += text_token
            sent_token += [SEP]
            segment_ids = [0] * len(sent_token)
            for i in range(1, len(sent)):
                text = sent[i]
                length = sent_length[i]
                text_token = self.tokenizer.tokenize(text)[:length-1] + [SEP]
                sent_token += text_token
                segment_ids += [1] * len(text_token)
        
        max_seq_length = sum(sent_length) + 1
        # sent_token to input_ids 
        input_ids = self.tokenizer.convert_tokens_to_ids(sent_token)
        input_mask = [1] * len(input_ids)

        # padding
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        if isinstance(sent, list) is True: 
            segment_ids += padding
            assert len(segment_ids) == max_seq_length 
        else:
            segment_ids = None
        assert len(input_mask) == max_seq_length
        assert len(input_ids) == max_seq_length

        # bert_input
        if segment_ids is None or input_mask is None:
            bert_input = {
                        'input_ids' : input_ids, 
                        'attention_mask' : input_mask,
                        'sent_token' : sent_token
                        }
        else:
            bert_input = {
                        'input_ids' : input_ids, 
                        'attention_mask' : input_mask, 
                        'sent_token' : sent_token
                        }
        return bert_input

                        #'token_type_ids' : segment_ids,

    def boxplot_for_sent_token_length(self, batch_sent=list, show=True):
        '''
        [DONE]
        '''
        token_len_list = []
        for sent in tqdm(batch_sent): 
            bert_input = self.convert_sent_to_token(sent=sent, max_seq_length=None)
            sent_token = bert_input['sent_token']
            token_len_list.append(len(sent_token))
        if show is True:
            avg = sum(token_len_list) / len(token_len_list)
            count = len(token_len_list)
            print('avg (token_len_list) : ', avg)
            print('var (token_len_list)', (sum([(val-avg)**2 for val in token_len_list]) ** (1/2)) / count)
            print('max (token_len_list) : ', max(token_len_list))
            print('min (token_len_list) : ', min(token_len_list))
            print('count (token_len_list) : ', count)
            plt.boxplot(token_len_list)
            plt.show()
        else:
            return token_len_list



    def transform_bert_input_into_tensor(self, bert_input=dict):
        for key in list(bert_input.keys()):
            bert_input[key] = torch.tensor(bert_input[key]).to(self.device)
        return bert_input
        

    def convert_bert_input_to_CLS(self, bert_input=dict, tensor=True, bert_model=None):
        # bert_input to bert_input-tensor
        bert_input = self.transform_bert_input_into_tensor(bert_input=bert_input)
        
        # bert_input to cls-rep
        if tensor is False:
            batch_cls_rep = np.array(bert_model(**bert_input)[-1].tolist())
        else:
            batch_cls_rep = bert_model(**bert_input)[1]
        
        return batch_cls_rep





if __name__ == '__main__':
    from transformers import AutoModelForSequenceClassification
    model_name = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name)







