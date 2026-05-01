import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import metrics

from . import data_reader
from . import dataset
from . import model
from . import utils


class TestRunner(utils.Runner):

    def __init__(self, args):
        args.config_path = f'{args.model_dir}/config.yaml'
        super().__init__(args)
        config = self.config
        config.update(args, ignore_none=True)

        input_path = args.input_path
        if input_path.endswith('.smi'):
            self.df = pd.read_csv(input_path, sep=' ', names=['smiles', 'name'])
            data = [list(self.df['smiles'])]
        elif input_path.endswith('.csv'):
            self.df = pd.read_csv(input_path)
            data = [input_path]
        config.is_pair_data = [args.is_pair_data]

        if config.model_name == 'generaldl':
            self.model_class = model.GeneralDL
            self.cal_metric = utils.cal_generaldl_metric
        elif config.model_name == 'specdl':
            self.model_class = model.SpecDL
            self.cal_metric = utils.cal_specdl_metric

        self.init_torch()
        self.data = data_reader.init_data(data, config)
        parent_path = os.path.dirname(config.output_path)
        if not parent_path:
            parent_path = '.'
        self.logger = utils.set_logger(f'{parent_path}/test_log.txt')

    def predict_ensemble(self):
        config = self.config
        model_dir = config.model_dir      

        final_pred = np.zeros((len(self.df),), dtype=float)
        # Trying with only 1 num fold to see difference and if process is faster
        num_folds = 1
        for i in range(num_folds):
            model_path = f'{model_dir}/model_{i}.pth'
            self.logger.info(f'Start predicting with model:{model_path}')
            X = [np.asarray(self.data[0]['features'])]
            y = [np.asarray(self.data[0]['target'])]
            test_subsets = []
            for i in range(len(X)):
                test_subsets.append(dataset.SubDataset(X[i], y[i], i))
            test_set = dataset.MultipleDataset(test_subsets, config)
            pred_list = self.predict(model_path, test_set, is_pair_data=config.is_pair_data)

            pred = pred_list[:, 0]
            final_pred += pred
        final_pred /= num_folds
        smile_cols = ['smiles'] if not config.is_pair_data[0] else ['smiles_i', 'smiles_j']
        if 'label' in self.df:
            df = self.df[smile_cols + ['label']]
            label = df['label']
            assert df['label'].isin([0, 1]).all()
            df['prediction'] = final_pred
            metric = metrics.roc_auc_score(label, final_pred)
            self.logger.info(f'test_auc: {metric}')
        else:
            df = self.df[smile_cols]
            df['prediction'] = final_pred
        df.to_csv(config.output_path, index=None)
        self.logger.info(f'Results saved in {config.output_path}')
    
    def predict(self, model_path, test_set, is_pair_data=False):
        config = self.config
        config.target_cols = ['label']
        config.is_pair_data = is_pair_data        

        model = self.model_class(config).to(self.device)
        self.logger.info(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.logger.info("Loaded weights successfully!")

        test_set_lens = [len(subset) for subset in test_set.subset_list]
        test_sampler = dataset.MultiDatasetPredictSampler(dataset_lens=test_set_lens)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size,
            shuffle=False, collate_fn=test_set.collate_fn, sampler=test_sampler)
        
        model = model.eval()
        prog_bar = tqdm(total=len(test_loader), dynamic_ncols=True,
                        position=0, leave=False, desc='test', ncols=5)
        test_preds = []
        for batch in test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, return_loss=False, return_pred=True)
                test_pred = outputs['predict'].cpu().numpy()
                test_preds.append(test_pred)
            prog_bar.update()
        test_preds = np.concatenate(test_preds)

        return test_preds
