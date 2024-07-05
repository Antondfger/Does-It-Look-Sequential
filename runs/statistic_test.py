"""
Run statistical difference test.
"""
import os
import sys

sys.path.append(os.environ['PATH4SEQ'])

import hydra
import pandas as pd
from clearml import Task
from omegaconf import OmegaConf

from stats.stat_test import stat_diff


@hydra.main(config_path="conf", config_name="statistic_test")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    stats = stat_diff(data_folder=config.datasets_info.path_to_metrics_by_user, dataset=config.datasets_info.name, 
                      metric_name=config.metric_name, alpha=config.alpha, n_iter=config.n_iter, seed=config.seed)

    print(stats)

    if task:
        
        clearml_logger = task.get_logger()
        clearml_logger.report_single_value('stat_diff', stats)

if __name__ == "__main__":

    main()
