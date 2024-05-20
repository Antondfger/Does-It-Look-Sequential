"""
Metrics.
"""

from recommenders.evaluation import python_evaluation


DEFAULT_METRICS = ['map_at_k', 'ndcg_at_k', 'recall_at_k', 'precision_at_k']
DEFAULT_METRICS_BEYOND_ACCURACY = ['catalog_coverage', 'distributional_coverage', 'novelty']

METRIC_NAMES = {
    'map_at_k': 'map',
    'ndcg_at_k': 'ndcg',
    'recall_at_k': 'recall',
    'precision_at_k': 'precision',
    'catalog_coverage': 'coverage',
    'distributional_coverage': 'entropy'
}


class Evaluator:
    """Class for computing recommendation metrics."""

    def __init__(self, metrics=DEFAULT_METRICS,
                 metrics_beyond_accuracy=DEFAULT_METRICS_BEYOND_ACCURACY,
                 top_k=[10], col_user='user_id', col_item='item_id',
                 col_prediction='prediction', col_rating='rating'):

        self.metrics = metrics
        self.metrics_beyond_accuracy = metrics_beyond_accuracy
        self.top_k = top_k
        self.col_user = col_user
        self.col_item = col_item
        self.col_prediction = col_prediction
        self.col_rating = col_rating

    def compute_metrics(self, test, recs, train=None):
        """Compute all metrics.

        Args:
            test (pd.DataFrame): Dataframe with test data.
            recs (pd.DataFrame): Dataframe with recommendations.
            train (pd.DataFrame): Dataframe with train data.
            
        Returns:
            Cosine similarity
        """

        if not hasattr(test, 'rating'):
            test = test.assign(rating=1)

        result = {}
        for k in self.top_k:
            for metric in self.metrics:
                metric_obj = getattr(python_evaluation, metric)
                metric_name = METRIC_NAMES.get(metric) or metric
                result[f'{metric_name}@{k}'] = metric_obj(
                    test, recs,  k=k, col_user=self.col_user, col_item=self.col_item,
                    col_prediction=self.col_prediction, col_rating=self.col_rating)
            if train is not None:
                for metric in self.metrics_beyond_accuracy:
                    metric_obj = getattr(python_evaluation, metric)
                    metric_name = METRIC_NAMES.get(metric) or metric
                    try:
                        result[f'{metric_name}@{k}'] = metric_obj(
                            train, recs, col_user=self.col_user, col_item=self.col_item)
                    except:
                        pass

        return result
