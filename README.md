# Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations
## Abstract
Sequential recommender systems are an important and demanded area of research. Such systems aim to use the order of interactions in a user’s history to predict future interactions. The premise is that the order of interactions and sequential patterns play an important role. Therefore, it is crucial to use datasets that exhibit a sequential structure for a proper evaluation of sequential recommenders. \
We apply several methods based on the random shuffling of the user's sequence of interactions to assess the strength of sequential structure across 15 datasets, frequently used for sequential recommender systems evaluation in recent research papers presented at top-tier conferences. As shuffling explicitly breaks sequential dependencies inherent in datasets, we estimate the strength of sequential patterns by comparing metrics for shuffled and original versions of the dataset. Our findings show that several popular datasets have a rather weak sequential structure.
## Main results
| Dataset      | HR@10 | NDCG@10 | Jaccard@10 | 2-grams | 3-grams |
|--------------|-------|---------|------------|---------|---------|
| Beauty       | -39 % | -43 %   | -74 %      | 0,03    | 0       |
| Diginetica   | -14 % | -7 %    | -6 %       | 0,26    | 0,06    |
| OTTO         | -30 % | -28 %   | -50 %      | 0,1     | 0,04    |
| RetailRocket | -4 %  | -2 %    | -29 %      | 0,46    | 0,33    |
| SMM          | -47 % | -45 %   | -44 %      | 0,02    | 0,02    |
| Sports       | -28 % | -32 %   | -60 %      | 0,06    | 0       |
| Yoochoose    | -22 % | -27 %   | -26 %      | 0,18    | 0,4     |
| Games        | -33 % | -38 %   | -63 %      | 0,08    | 0,02    |
| Steam        | -10 % | -12 %   | -6 %       | 0       | 0,01    |
| ML-20m       | -59 % | -61 %   | -59 %      | 0       | 0       |
| 30Music      | -90 % | -92 %   | -59 %      | 0       | 0       |
| Zvuk         | -68 % | -70 %   | -74 %      | 0,01    | 0       |
| Foursquare   | -7 %  | -5 %    | -59 %      | 0,42    | 0,22    |
| Gowalla      | -8 %  | -8 %    | 54 %       | 0,44    | 0,18    |
| Yelp         | -2 %  | 5 %     | -2 %       | 0,05    | 0       |

## Usage
Install requirements:
```sh
pip install -r requirements.txt
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
