# Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations
## Abstract
Sequential recommender systems are an important and demanded area of research. Such systems aim to use the order of interactions in a user’s history to predict future interactions. The premise is that the order of interactions and sequential patterns play an important role. Therefore, it is crucial to use datasets that exhibit a sequential structure for a proper evaluation of sequential recommenders. \
We apply several methods based on the random shuffling of the user's sequence of interactions to assess the strength of sequential structure across 15 datasets, frequently used for sequential recommender systems evaluation in recent research papers presented at top-tier conferences. As shuffling explicitly breaks sequential dependencies inherent in datasets, we estimate the strength of sequential patterns by comparing metrics for shuffled and original versions of the dataset. Our findings show that several popular datasets have a rather weak sequential structure.
## Main results
|---------------|---------|---------|----------|---------|---------|----------|---------|---------|----------|---------|---------|----------|---------|---------|----------|
| Datasets      | Before  | After   | Relative | Before  | After   | Relative | Before  | After   | Relative | Before  | After   | Relative | Before  | After   | Relative |
|              | shuffle | shuffle | change   | shuffle | shuffle | change   | shuffle | shuffle | change   | shuffle | shuffle | change   | shuffle | shuffle | change   |
|---------------|---------|---------|----------|---------|---------|----------|---------|---------|----------|---------|---------|----------|---------|---------|----------|
| Beauty        | 443     | 11,6    | -97\%    | 131     | 0,0     | -100\%   | 0,042   | 0,026   | -39\%    | 0,019   | 0,011   | -43\%    | 0,94    | 0,24    | -74\%    |
| Diginetica    | 1464    | 379,4   | -74\%    | 64      | 3,6     | -94\%    | 0,333   | 0,286   | -14\%    | 0,161   | 0,149   | -7\%     | 0,55    | 0,52    | -6\%     |
| OTTO          | 4907    | 491,8   | -90\%    | 1942    | 69,0    | -96\%    | 0,205   | 0,143   | -30\%    | 0,120   | 0,086   | -28\%    | 0,56    | 0,28    | -50\%    |
| RetailRocket  | 2463    | 1145,0  | -54\%    | 730     | 238,4   | -67\%    | 0,326   | 0,315   | -4\%     | 0,195   | 0,190   | -2\%     | 0,66    | 0,47    | -29\%    |
| MegaMarket    | 54265   | 1163,6  | -98\%    | 35775   | 851,4   | -98\%    | 0,192   | 0,101   | -47\%    | 0,111   | 0,062   | -45\%    | 0,34    | 0,19    | -44\%    |
| Sports        | 129     | 7,4     | -94\%    | 16      | 0,0     | -100\%   | 0,032   | 0,023   | -28\%    | 0,016   | 0,011   | -32\%    | 0,67    | 0,26    | -60\%    |
| Yoochoose     | 16848   | 3057,2  | -82\%    | 73488   | 29528,2 | -60\%    | 0,396   | 0,308   | -22\%    | 0,228   | 0,167   | -27\%    | 0,63    | 0,46    | -26\%    |
| Games         | 207     | 17,4    | -92\%    | 19      | 0,4     | -98\%    | 0,052   | 0,035   | -33\%    | 0,025   | 0,015   | -38\%    | 0,61    | 0,22    | -63\%    |
| Steam         | 247     | 1,2     | -100\%   | 335     | 3,8     | -99\%    | 0,110   | 0,099   | -10\%    | 0,053   | 0,047   | -12\%    | 0,63    | 0,59    | -6\%     |
| ML-20m        | 536     | 0,0     | -100\%   | 46727   | 4,8     | -100\%   | 0,075   | 0,031   | -59\%    | 0,036   | 0,014   | -61\%    | 0,29    | 0,12    | -59\%    |
| 30Music       | 273654  | 463,6   | -100\%   | 267817  | 149,2   | -100\%   | 0,198   | 0,020   | -90\%    | 0,136   | 0,010   | -92\%    | 0,29    | 0,12    | -59\%    |
| Zvuk          | 77209   | 414,0   | -99\%    | 154009  | 172,4   | -100\%   | 0,216   | 0,069   | -68\%    | 0,112   | 0,034   | -70\%    | 0,42    | 0,11    | -74\%    |
| Foursquare    | 3937    | 1641,0  | -58\%    | 3628    | 797,4   | -78\%    | 0,353   | 0,328   | -7\%     | 0,224   | 0,213   | -5\%     | 0,94    | 0,39    | -59\%    |
| Gowalla       | 26181   | 11486,0 | -56\%    | 9012    | 1635,0  | -82\%    | 0,301   | 0,277   | -8\%     | 0,186   | 0,170   | -8\%     | 0,29    | 0,45    | +54\%    |
| Yelp          | 30      | 1,6     | -95\%    | 4       | 0,0     | -100\%   | 0,044   | 0,043   | -2\%     | 0,021   | 0,022   | +5\%     | 0,38    | 0,37    | -2\%     |

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
