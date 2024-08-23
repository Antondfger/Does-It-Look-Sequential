import numpy as np
import pandas as pd
from scipy.stats import norm


def check_hyp_equality(df_pilot_group, df_control_group, metric_name, n_iter, alpha=0.05, seed=None):
    
    np.random.seed(seed)
    bootstrap_pilot = np.random.choice(df_pilot_group[metric_name], size=(len(df_pilot_group), n_iter))
    bootstrap_control = np.random.choice(df_control_group[metric_name], size=(len(df_control_group), n_iter))
    bootstrap = bootstrap_pilot.mean(1) - bootstrap_control.mean(1)
    mean = float(df_pilot_group[metric_name].mean() - df_control_group[metric_name].mean())
    res = mean+float(norm.ppf(alpha/2)*bootstrap.std())<=0<=mean+float(norm.ppf(1-alpha/2)*bootstrap.std())
    
    return res, mean


def stat_diff(data_folder, dataset, name, metric_name, n_iter, alpha=0.05, seed=None):
    """Shows statistical differences between user metrics for shuffle and non-shuffle data.
    
    Args:
        data_folder (str): Folder with user metrics.
        dataset (str): Dataset name.
        metric_name (str): Metric name.
        name (str): type_model
        alpha (float): Significance level.
        n_iter (str): Number of bootstrap samples.
        seed (int): Defaults to None.
       
    Returns:
        True if there are no statistically significant differences, false if there are and confidence interval.
    """
    shuffle = pd.read_csv(f"{data_folder}/{name}_{dataset}_shuffle.csv")
    origin = pd.read_csv(f"{data_folder}/{name}_{dataset}_origin.csv")
    res, mean = check_hyp_equality(origin,shuffle,metric_name,n_iter=n_iter,seed=seed,alpha=alpha)

    return res, mean
