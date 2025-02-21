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
        core = config.download_core
        data = pd.read_csv(path_to_split + 'test_' +  f'core_{core}_' + config.datasets_info.name + '.csv')
        data = pd.read_csv(config.datasets_info.data_path)
        data = rename(raw_data, **config.datasets_info.column_name)
        data_path = config.datasets_info.data_path
        data = pd.read_csv(data_path)
     

    else:
        core = str(min(config.prepr.prep_params.min_len, config.prepr.prep_params.item_min_count))
        data = preprocessing(raw_data, **config.prepr.prep_params)
   
    stats = statistics(data)
    
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