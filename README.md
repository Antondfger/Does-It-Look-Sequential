# Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations
## Abstract
Sequential recommender systems are an important and demanded area of research. Such systems aim to use the order of interactions in a user’s history to predict future interactions. The premise is that the order of interactions and sequential patterns play an important role. Therefore, it is crucial to use datasets that exhibit a sequential structure for a proper evaluation of sequential recommenders.
We apply several methods based on the random shuffling of the user's sequence of interactions to assess the strength of sequential structure across 15 datasets, frequently used for sequential recommender systems evaluation in recent research papers presented at top-tier conferences. As shuffling explicitly breaks sequential dependencies inherent in datasets, we estimate the strength of sequential patterns by comparing metrics for shuffled and original versions of the dataset. Our findings show that several popular datasets have a rather weak sequential structure.
## Main results
----------------
## Usage
For configuration we use [Hydra](https://hydra.cc/). Parameters are specified in [config files](src/configs/), they can be overriden from the command line. Optionally it is possible to use [ClearML](`https://clear.ml/docs/latest/docs`) for experiments logging (`project_name` and `task_name` should be specified in config to use ClearML).

Example of run via command line:
```sh
cd runs
python dl.py datasets_info=Movielens
```
### Reproduce paper results
Scripts to reproduce SRS's results: dl.sh \
Scripts to reproduce the strength of sequential patterns: rule.sh \
Scripts to reproduce dataset statistics: statistics.sh

```sh
cd runs
sh dl.sh
sh rule.sh
sh statistics.sh
```








#### In the [datasets](datasets) folder, you will find detailed information about the process of selecting datasets for this project.
