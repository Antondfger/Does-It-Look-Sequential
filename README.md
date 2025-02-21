# Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations
This repository contains code for extended version of ACM RecSys 2024 paper ["Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations?"](
https://doi.org/10.48550/arXiv.2408.12008)
## Abstract
Sequential recommender systems are an important and demanded area of research. Such systems aim to use the order of interactions in a user’s history to predict future interactions. The premise is that the order of interactions and sequential patterns play an important role. Therefore, it is crucial to use datasets that exhibit a sequential structure for a proper evaluation of sequential recommenders. \
We apply several methods based on the random shuffling of the user's sequence of interactions to assess the strength of sequential structure across 15 datasets, frequently used for sequential recommender systems evaluation in recent research papers presented at top-tier conferences. As shuffling explicitly breaks sequential dependencies inherent in datasets, we estimate the strength of sequential patterns by comparing metrics for shuffled and original versions of the dataset. Our findings show that several popular datasets have a rather weak sequential structure.

## Main results
In this paper, we proposed a set of three approaches to evaluate a dataset's sequential structure strength. We further analyzed a wide range of datasets from different domains that are commonly used for the evaluation of SRSs. The results of our experiments show that many popular datasets, namely Diginetica, Foursquare, Gowalla, RetailRocket, Steam, and Yelp, lack a sequential structure.

## Usage
Install requirements:
```sh
pip install -r requirements.txt
```
Specify environment variables:
```sh
# path to the project
export PATH4SEQ="/your/path"
# path to the raw data
export RECSYS_DATA_PATH="/your/path"
# path where the data will be stored after preprocessing
export PREP_DATA_PATH="/your/path"
# path where the data will be stored after split
export SPLIT_DATA_PATH="/your/path"
```

For configuration we use [Hydra](https://hydra.cc/). Parameters are specified in [config files](runs/conf/), they can be overriden from the command line. Optionally it is possible to use [ClearML](`https://clear.ml/docs/latest/docs`) for experiments logging (`project_name` and `task_name` should be specified in config to use ClearML).

Example of run via command line:
```sh
cd runs
python dl.py datasets_info=Movielens-20
```
## Reproduce paper results
Scripts to reproduce 2-core results: 2_core.sh \
Scripts to reproduce 5-core results: 5_core.sh \
Scripts to reproduce 10-core results: 10_core.sh \
Scripts to reproduce shuffle in training results: 5_core-shuffle.sh \
Scripts to reproduce sequential rules: rule.sh \
Scripts to reproduce dataset statistics: statistics.sh


```sh
cd runs
sh 2_core.sh
sh 5_core.sh
sh 10_core.sh
sh 5_core-shuffle.sh
sh rule.sh
sh statistics.sh
```
## Datasets selection
 In the [datasets](datasets) folder, you will find detailed information about the process of selecting datasets for this project.
