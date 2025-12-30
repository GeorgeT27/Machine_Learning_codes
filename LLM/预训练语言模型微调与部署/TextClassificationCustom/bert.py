from transformers import BertpreTrainedModel,BertModel
from transfomers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
class BertForSequenceClassification(BertpreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels=config.num_labels
        self.config=config
        self.bert=BertModel(config)
        classifier_dropout=(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.linear(config.hiddent_size,config.num_labels)
        self.post_init()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        '''
        :param input_ids: interger ids where each id maps to a token in the tokenizer vocabulary 
        we are using pretrained BERT tokenizer,so the input_ids should be compatible with BERT tokenizer
        
        :param attention_mask: Make sure that the model does not attend to padding tokens 
        since we would like each length of input to be same in a batch, we pad the shorter sequences with zeros
        
        :param token_type_ids: give 0/1 to differentiate two sentences in tasks. e.g, question answering where we 
        select answers from the context given the questions
        
        :param position_ids: Transformers are position-agnostic by default, but we could customimise it by providing position_ids
        
        :param head_mask: a mask over attention heads for analysis purposes
        
        :param inputs_embeds: we could directly pass the embeddings instead of input_ids
        
        :param labels: Ground-truth targets.
        
        :param output_attentions: whether to return the attentions weights.
        
        :param output_hidden_states: whether to return the hidden states.
        
        :param return_dict: whether to return a dict instead of a tuple, so functions such as outputs.hidden_states can be used.
        '''
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict
        outputs=self.bert(
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
        pooled_output=outputs[1]
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output) 
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
                loss_fct=nn.MSELoss()
                if self.num_labels==1:
                    loss=loss_fct(logits.squeeze(),labels.squeeze())
                else:
                    loss=loss_fct(logits,labels)
            elif self.config.problem_type=="single_label_classification":
                loss_fct=nn.CrossEntropyLoss()
                loss=loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            elif self.config.problem_type=="multi_label_classification":
                loss_fct=nn.BCEWithLogitsLoss()
                loss=loss_fct(logits,labels)
        if not return_dict:
            output=(logits,)+outputs[2:]
            return ((loss,)+output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        