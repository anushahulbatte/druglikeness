from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import load_state_dict, _load_state_dict_into_model
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import os 

logger = logging.getLogger('utils')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pretrained_path = os.path.join(base_dir, 'weights', 'chemberta')

# pretrained_path = 'weights/chemberta'

dln_activation_dict = {'linear': lambda x: x,
                       'tanh': torch.tanh,
                       'sigmoid': torch.sigmoid,
                       'relu': torch.relu}

class DruglikenessModel(RobertaPreTrainedModel):
    def load_pretrained_weights(self, model_path):
        print(f'loading pretrained model from {model_path}')
        state_dict = load_state_dict(model_path)
        expected_keys = list(self.state_dict().keys())
        loaded_keys = list(state_dict.keys())       
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        error_msgs = _load_state_dict_into_model(self, state_dict, start_prefix='')
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{error_msg}")
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {model_path} were not used when"
                f" initializing {self.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {self.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {self.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {self.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {self.__class__.__name__} were not initialized from the model checkpoint at"
                f" {model_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_tasks)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def focal_loss(y_pred, y_real, alpha=0.25, gamma=2.0):
    if y_pred.shape != y_real.shape:
        y_real = y_real.flatten()
    y_real = y_real.long()
    y_pred = y_pred.float()
    y_real = y_real.float()
    y_real = y_real.unsqueeze(1)
    y_pred = y_pred.unsqueeze(1)
    y_real = torch.cat((1-y_real, y_real), dim=1)
    y_pred = torch.cat((1-y_pred, y_pred), dim=1)
    y_pred = y_pred.clamp(1e-5, 1.0)
    loss = -alpha * y_real * torch.pow((1 - y_pred), gamma) * torch.log(y_pred)
    return torch.mean(torch.sum(loss, dim=1))


class GeneralDL(DruglikenessModel):
    def __init__(self, config):
        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_path, local_files_only = True)
        roberta_config = RobertaConfig.from_dict(config_dict)
        super().__init__(roberta_config)

        roberta_config.num_tasks = config.num_tasks
        self.dln_weight = config.dln_weight
        self.pair_subsets = [index for index, value in enumerate(config.is_pair_data) if value]
        self.roberta = RobertaModel(roberta_config, add_pooling_layer=False)
        self.classifier = ClassificationHead(roberta_config)
        self.druglikeness_head = nn.Linear(config.num_tasks, 2)
        
        self.dln_activation_fn = dln_activation_dict[config.dln_activation]

        self.post_init()

        if config.load_pretrained_weights:
            self.load_pretrained_weights(f'{pretrained_path}/pytorch_model.bin')
        else:
            print('Not Loading Pretrained weights. Start with random init.')

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, 
                target: torch.LongTensor, subset: torch.LongTensor, 
                return_loss=True, return_pred=False): 
               
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        raw_mt_logits = self.classifier(sequence_output)
        x = self.dln_activation_fn(raw_mt_logits)
        raw_dln_logits = self.druglikeness_head(x)

        # Changing from cude:0 to cpu 
        mt_logits = torch.full(target.shape, torch.nan, device='cpu')
        dln_logits = torch.full((target.shape[0], 2), torch.nan, device='cpu')
        i = 0
        for j, subset_idx in enumerate(subset):
            if subset_idx in self.pair_subsets:
                mt_logits[j] = raw_mt_logits[i+1] - raw_mt_logits[i]
                dln_logits[j] = raw_dln_logits[i+1] - raw_dln_logits[i]
                i += 2
            else:
                mt_logits[j] = raw_mt_logits[i]
                dln_logits[j] = raw_dln_logits[i]
                i += 1

        results = {}
        if return_loss:
            results['loss'] = self.loss_fn(mt_logits, dln_logits, target)
        if return_pred:
            results['predict'] = F.softmax(dln_logits, dim=-1)[:, 1:]
        return results

    def loss_fn(self, mt_logits, dln_logits, target):
        dln_weight = self.dln_weight
        mt_pred = torch.sigmoid(mt_logits)
        mask = ~torch.isnan(target)
        target = target[mask]
        mt_pred = mt_pred[mask]

        dln_mask = ~torch.isnan(dln_logits).any(dim=1)
        dln_logits = dln_logits[dln_mask]
        dln_true = target[dln_mask]

        loss_mt = focal_loss(mt_pred, target)
        loss_dln = nn.CrossEntropyLoss()(dln_logits, dln_true.long())
        loss = loss_mt * (1 - dln_weight) + loss_dln * dln_weight
        return loss
    

class SpecDL(DruglikenessModel):
    def __init__(self, config):
        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_path)
        roberta_config = RobertaConfig.from_dict(config_dict)

        super().__init__(roberta_config)
        roberta_config.num_tasks = config.num_tasks
        self.pair_subsets = [index for index, value in enumerate(config.is_pair_data) if value]
        self.roberta = RobertaModel(roberta_config, add_pooling_layer=False)
        self.classifier = ClassificationHead(roberta_config)
        self.post_init()

        if config.load_pretrained_weights:
            self.load_pretrained_weights(f'{pretrained_path}/pytorch_model.bin')
        else:
            print('Not Loading Pretrained weights. Start with random init.')

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, 
                target: torch.LongTensor, subset: torch.LongTensor, 
                return_loss=True, return_pred=False):

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        raw_logits = self.classifier(sequence_output)
        
        # Changing device from cude:0 to cpu 
        logits = torch.full(target.shape, torch.nan, device='cpu')
        i = 0
        for j, subset_idx in enumerate(subset):
            if subset_idx in self.pair_subsets:
                logits[j] = raw_logits[i+1] - raw_logits[i]
                i += 2
            else:
                logits[j] = raw_logits[i]
                i += 1

        results = {}
        if return_loss:
            results['loss'] = self.loss_fn(logits, target)
        if return_pred:
            results['predict'] = torch.sigmoid(logits)
        return results

    def loss_fn(self, logits, target):
        pred = torch.sigmoid(logits)
        mask = ~torch.isnan(target)
        pred = pred[mask]
        target = target[mask]
        loss = focal_loss(pred, target)
        return loss
