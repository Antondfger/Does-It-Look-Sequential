"""
Run sequential rules counter.
"""
import os
import sys
sys.path.append(os.environ['PATH4SEQ'])

import pandas as pd
import hydra
from omegaconf import OmegaConf
from clearml import Task
from stats.rules import rule_counter
from preprocessing.preprocessing import preprocessing


@hydra.main(config_path="conf", config_name="rules")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    if config.download_data:
        path_to_split = config.datasets_info.path_to_prep_data
        data = pd.read_csv(path_to_split)

    else:
        data = pd.read_csv(config.datasets_info.data_path)
        data = preprocessing(data, **config.prepr.prep_params, **config.datasets_info.column_name)

    rule = rule_counter(data, **config.rule_params, random_state=config.random_state).to_frame().T

    print(rule)

    if task:

        clearml_logger = task.get_logger()

        for key, value in rule.items():
            clearml_logger.report_single_value(key, value)

        clearml_logger.report_table(title='rule_metrics', series='dataframe',
                                    table_plot=pd.DataFrame(rule))
        task.upload_artifact('dataset_metrics', pd.DataFrame(rule))

if __name__ == "__main__":

    main()
