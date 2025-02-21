"""
Metrics.
"""
import inspect
import pandas as pd
from replay import metrics as base_class
from replay.metrics import OfflineMetrics


DEFAULT_METRICS = ['NDCG', 'HitRate', 'Precision', 'Recall', 'MRR']

class Evaluator:
    """Class for computing recommendation metrics.
    """

    def __init__(self, metrics=DEFAULT_METRICS, topk=[10, 100], modes=['Mean'], user_id='user_id', item_id='item_id',                            rating_columns='prediction'):
        """Args:
            metrics (list): List with metrics name. The names are taken from the Replay.
            topk (list): Consider the highest k scores in the ranking. Defaults to [10, 100].
            modes (list): Classes for calculating aggregation metrics. Defaults to Mean. Available modes: Median,         ConfidenceInterval, PerUser.
            user_id (str): Defaults to 'user_id'.
            item_id (str): Defaults to 'item_id'.
            rating_columns (str): Defaults to 'rating'."""
        
        self.metrics = metrics
        self.topk = topk
        self.modes = modes
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating_columns
        
        class_method = [x[0] for x in inspect.getmembers(base_class)[:19]]

        if type(metrics) != list:
            raise ValueError("Use the list data type for metrics.")
        
        if type(topk) != list:
            raise ValueError("Use the list data type for topk.")
        
        if type(modes) != list:
            raise ValueError("Use the list data type for modes.")
        
        if len(modes) > 1 and 'PerUser' in modes:
            raise ValueError("Mode 'PerUser' can use only alone.")

        for mode in modes:
            if mode not in class_method:
                raise ValueError(f"{mode} is not available in Replay. Look at the documentation. https://sb-ai-                                                      lab.github.io/RePlay/pages/modules/metrics.html#replay.metrics")
            
        for metric in metrics:
            if metric not in class_method:
                raise ValueError(f"{metric} is not available in Replay. Look at the documentation. https://sb-ai-                                                    lab.github.io/RePlay/pages/modules/metrics.html#replay.metrics")
         
    def compute_metrics(self, test, recs, train=None):
        """Compute all metrics.

        Args:
            test (pd.DataFrame): Dataframe with test data.
            recs (pd.DataFrame): Dataframe with recommendations.
            train (pd.DataFrame): Dataframe with train data.
            
        Returns:
            metrics
        """
        
        metrics_list = []
        for metric in self.metrics:
            for k in self.topk:
                for mode in self.modes:
                    mode = getattr(base_class, mode)()
                    metrics_list.append(getattr(base_class, metric)(topk=k, mode=mode))
        
        metrics = OfflineMetrics(metrics_list, query_column=self.user_id, 
                                 item_column=self.item_id, rating_column=self.rating)(recs, test, train)
        metrics = pd.DataFrame.from_dict(metrics, orient='index').T

        return metrics
