# 66666
This is an open source implementation of the baselines presented in the [Amazon Product Search KDD CUP 2022](https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search).


## Requirements
We launched the baselines experiments creating an environment with Python 3.6 and installing the packages  shown below:
```
pip3 install requirement.txt
```

## Download data

Before to launch the script below, it would be necessary to login in [aicrowd](https://www.aicrowd.com/) using the Python client `aicrowd login`.

The script below downloads all the files for the three tasks using the aicrowd client.

```bash
cd data/
bash download-data.sh
```
Note : After download data, please move these data to `task1, task2`.


### Task 1 - Query Product Ranking (DownStream)

For task 1, we fine-tuned `roberta-xlm` models for each `query_locale`.

 We used the query and title of the product as input for these models.

```
python3 main.py
```

### Task 1 - Query Product Ranking (UpStream)

For task 1, we pre-train `roberta-xlm` models for each `product_title + product_bullet_point`.

 We used the query and title of the product as input for these models.

```
python3 maskLMing.py
```
BTW, we not yet submit pretrain model to leaderboard.



## Results
The following table shows the baseline results obtained through the different public tests of Task1.

| Model |  Metric  | Score |
|:----:|:--------:|:-----:|
|   Roberta-XLM  | nDCG     | 0.874 |
|   Roberta-XLM-TwoStage  | nDCG     | 0.884 |
|    Official Baseline | nDCG | 0.850 |
|    - | - | - |


