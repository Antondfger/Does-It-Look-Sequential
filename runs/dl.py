"""
Run experiment.
"""
import os
import time
import sys
sys.path.append(os.environ['PATH4SEQ'])
os.environ["WORLD_SIZE"] = "1"

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import random
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch.utils.data import DataLoader

from nn.datasets import (CausalLMDataset, CausalLMPredictionDataset,
                         PaddingCollateFn)
from nn.metrics import Evaluator
from nn.models import SASRec, GRU4Rec
from nn.modules import SeqRec, SeqRecWithSampling
from nn.postprocess import preds2recs
from preprocessing.preparation import get_last_item, remove_last_item, shuffle
from preprocessing.preprocessing import preprocessing
from preprocessing.splitter import session_split
from stats.jaccard import jaccard_similarity


@hydra.main(config_path="conf", config_name="training")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    if config.download_data:
        path_to_split = config.datasets_info.path_to_split_data
        train = pd.read_csv(path_to_split + 'train_' + config.datasets_info.name + '.csv')
        test = pd.read_csv(path_to_split + 'test_' + config.datasets_info.name + '.csv')
        validation = pd.read_csv(path_to_split + 'validation_' + config.datasets_info.name + '.csv')
        max_item_id = max(train.item_id.max(), test.item_id.max(), validation.item_id.max())

    else:
        data = pd.read_csv(config.datasets_info.data_path)
        data = preprocessing(data, **config.prepr.prep_params, **config.datasets_info.column_name)
        train, validation, test = session_split(data, **config.splitter.split_params)
        max_item_id = data.item_id.max()

    seed = config.random_state
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, item_count=max_item_id)
    start_time = time.time()
    trainer, seqrec_module, model = training(model, train_loader, eval_loader, config)
    training_time = time.time() - start_time
    print('training_time', training_time)

    val_inputs = remove_last_item(validation)
    val_last_item = get_last_item(validation)
    recs_validation = predict(trainer, seqrec_module, val_inputs, config)
    evaluate(recs_validation, val_last_item, task, config, prefix='val')

    test_inputs = remove_last_item(test)
    test_last_item = get_last_item(test)
    recs_test = predict(trainer, seqrec_module, test_inputs, config)
    evaluate(recs_test, test_last_item, task, config, prefix='test')

    if config.shuffle_inference:
        shuffle_test_inputs = shuffle(test_inputs, config.random_state)
        shuffle_recs_test = predict(trainer, seqrec_module, shuffle_test_inputs, config)
        evaluate(shuffle_recs_test, test_last_item, task, config, prefix='shuffle_inf')

    if config.jaccard and config.shuffle_inference:
        sim = pd.Series()
        sim['jaccard'] = jaccard_similarity(recs_test, shuffle_recs_test)
        print(sim['jaccard'])

        if task:
            clearml_logger = task.get_logger()
            clearml_logger.report_single_value('jaccard', sim['jaccard'] )
            task.upload_artifact('test_pred.csv', recs_test)
            task.upload_artifact('shuffle_pred.csv', shuffle_recs_test)


def create_dataloaders(train, validation, config):

    train_dataset = CausalLMDataset(train,  **config['dataset_params'])
    eval_dataset = CausalLMPredictionDataset(
                    validation, max_length=config.dataset_params.max_length, validation_mode=True, )

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(config, item_count):

    if hasattr(config.dataset_params, 'num_negatives') and config.dataset_params.num_negatives:
        add_head = False
    else:
        add_head = True

    if config.model.model=='SASRec':
       model = SASRec(item_num=item_count, add_head=add_head, **config.model.model_params)
    
    if config.model.model=='GRU4Rec':
        model = GRU4Rec(vocab_size=item_count + 1, add_head=add_head,
                    rnn_config=config.model.model_params)


    return model


def training(model, train_loader, eval_loader, config):

    if hasattr(config.dataset_params, 'num_negatives') and config.dataset.num_negatives:
        seqrec_module = SeqRecWithSampling(model, **config['seqrec_module'])
    else:
        seqrec_module = SeqRec(model, **config['seqrec_module'])

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])

    trainer.fit(model=seqrec_module,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader)

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module, model


def predict(trainer, seqrec_module, data, config):

    predict_dataset = CausalLMPredictionDataset(
        data, max_length=config.dataset_params.max_length)

    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.top_k_metrics)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)

    return recs


def evaluate(recs, test_last, task, config, prefix='test'):

    all_metrics = {}

    for k in config.top_k_metrics:
        evaluator = Evaluator(top_k=[k])
        metrics = evaluator.compute_metrics(test_last, recs)
        metrics = {prefix + '_' + key: value for key, value in metrics.items()}
        all_metrics.update(metrics)

    if task:

        clearml_logger = task.get_logger()

        for key, value in all_metrics.items():
            clearml_logger.report_single_value(key, value)
        all_metrics = pd.Series(all_metrics).to_frame().reset_index()
        all_metrics.columns = ['metric_name', 'metric_value']

        clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
                                    table_plot=all_metrics)
        task.upload_artifact(f'{prefix}_metrics', all_metrics)


if __name__ == "__main__":

    main()