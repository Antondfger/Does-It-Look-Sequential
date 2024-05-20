"""
Compute dataset statistics.
"""
import os
import sys

sys.path.append(os.environ['PATH4SEQ'])

import hydra
import pandas as pd
from clearml import Task
from omegaconf import OmegaConf

from preprocessing.preprocessing import preprocessing, rename
from preprocessing.splitter import session_split
from stats.data_statistics import statistics


@hydra.main(config_path="conf", config_name="statistics")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    raw_data = pd.read_csv(config.datasets_info.data_path)
    raw_data = rename(raw_data, **config.datasets_info.column_name)

    if config.download_data:
        path_to_split = config.datasets_info.path_to_split_data
        train = pd.read_csv(path_to_split + 'train_' + config.datasets_info.name + '.csv')
        test = pd.read_csv(path_to_split + 'test_' + config.datasets_info.name + '.csv')
        validation = pd.read_csv(path_to_split + 'validation_' + config.datasets_info.name + '.csv')

        path_to_prep = config.datasets_info.path_to_prep_data
        data = pd.read_csv(path_to_prep)

    else:
        data = preprocessing(raw_data, **config.prepr.prep_params,
                             path_to_save_prep=config.datasets_info.path_to_prep_data)
        train, validation, test = session_split(
            data, **config.splitter.split_params,
            path_to_save_split=config.datasets_info.path_to_split_data,
            name=config.datasets_info.name)

    stats = statistics(data, raw_data,  train, test, config)

    print(stats)

    if task:

        clearml_logger = task.get_logger()

        for key, value in stats.items():
            clearml_logger.report_single_value(key, value)

        clearml_logger.report_table(title='dataset_metrics', series='dataframe',
                                    table_plot=pd.DataFrame(stats))
        task.upload_artifact('dataset_metrics', pd.DataFrame(stats))


if __name__ == "__main__":

    main()
