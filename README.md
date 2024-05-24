# Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations
## Abstract
Sequential recommender systems are an important and demanded area of research. Such systems aim to use the order of interactions in a user’s history to predict future interactions. The premise is that the order of interactions and sequential patterns play an important role. Therefore, it is crucial to use datasets that exhibit a sequential structure for a proper evaluation of sequential recommenders. \
We apply several methods based on the random shuffling of the user's sequence of interactions to assess the strength of sequential structure across 15 datasets, frequently used for sequential recommender systems evaluation in recent research papers presented at top-tier conferences. As shuffling explicitly breaks sequential dependencies inherent in datasets, we estimate the strength of sequential patterns by comparing metrics for shuffled and original versions of the dataset. Our findings show that several popular datasets have a rather weak sequential structure.
## Main results
In this paper, we proposed a set of three approaches to evaluate a dataset's sequential structure strength. We further analyzed a wide range of datasets from different domains that are commonly used for the evaluation of SRSs. The results of our experiments show that many popular datasets, namely Diginetica, Foursquare, Gowalla, RetailRocket, Steam, and Yelp, lack a sequential structure.
Whether these datasets are suitable for evaluating sequential recommendations is questionable and needs further research.

The datasets selected for evaluation must be aligned with the task at hand. Conclusions drawn about the relative performance of different algorithms may change after selecting more appropriate datasets. Whether this is true or not is a possible future research direction, as well as further investigation of approaches to the assessment of sequential structure in datasets.

On this table, you can observe the variation in key metrics expressed in percentages according to the formula: - (1 - metrics after shuffle / metrics before shuffle) * 100%.

| Dataset      | HR@10 | NDCG@10 | Jaccard@10 | 2-grams | 3-grams |
|--------------|-------|---------|------------|---------|---------|
| Beauty       | -39%  | -43%    | -74%       | -97%    | -100%   |
| Diginetica   | -14%  | -7%     | -6%        | -74%    | -94%    |
| OTTO         | -30%  | -28%    | -50%       | -90%    | -96%    |
| RetailRocket | -4%   | -2%     | -29%       | -54%    | -67%    |
| SMM          | -47%  | -45%    | -44%       | -98%    | -98%    |
| Sports       | -28%  | -32%    | -60%       | -94%    | -100%   |
| Yoochoose    | -22%  | -27%    | -26%       | -82%    | -60%    |
| Games        | -33%  | -38%    | -63%       | -92%    | -98%    |
| Steam        | -10%  | -12%    | -6%        | -100%   | -99%    |
| ML-20m       | -59%  | -61%    | -59%       | -100%   | -100%   |
| 30Music      | -90%  | -92%    | -59%       | -100%   | -100%   |
| Zvuk         | -68%  | -70%    | -74%       | -99%    | -100%   |
| Foursquare   | -7%   | -5%     | -59%       | -58%    | -78%    |
| Gowalla      | -8%   | -8%     | 54%        | -56%    | -82%    |
| Yelp         | -2%   | 5%      | -2%        | -95%    | -100%   |

## Usage
Install requirements:
```sh
pip install -r requirements.txt
```
Specify environment variables. Where PATH4SEQ is the path to the project, RECSYS_DATA_PATH is the path to the raw data, PREP_DATA_PATH is the path where the data will be stored after preprocessing.
```sh
export RECSYS_DATA_PATH="${RECSYS_DATA_PATH}/your/path"

export PREP_DATA_PATH="${PREP_DATA_PATH}/your/path"

export SPLIT_DATA_PATH="${SPLIT_DATA_PATH}/your/path"
```

For configuration we use [Hydra](https://hydra.cc/). Parameters are specified in [config files](runs/conf/), they can be overriden from the command line. Optionally it is possible to use [ClearML](`https://clear.ml/docs/latest/docs`) for experiments logging (`project_name` and `task_name` should be specified in config to use ClearML).

Example of run via command line:
```sh
cd runs
python dl.py datasets_info=Movielens
```
## Reproduce paper results
Scripts to reproduce SRS's results: dl.sh \
Scripts to reproduce the strength of sequential patterns: rule.sh \
Scripts to reproduce dataset statistics: statistics.sh

```sh
cd runs
sh dl.sh
sh rule.sh
sh statistics.sh
```
## Datasets selection
 In the [datasets](datasets) folder, you will find detailed information about the process of selecting datasets for this project.
