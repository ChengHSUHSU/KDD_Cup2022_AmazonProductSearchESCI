# KDD CUP 2022 AmazonProductSearch ESCI
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

| Model |  Metric  | Online-Score |
|:----:|:--------:|:-----:|
|   (1) Roberta-XLM (MSE)  | nDCG     | 0.874 |
|   (2) Roberta-XLM (MSE-TwoStage)  | nDCG     | 0.884 |
|   (3) Deberta-Large-V3 + Roberta-XLM (MSE-TwoStage)  | nDCG     | 0.894 |
|   (4) Official Baseline | nDCG | 0.850 |
|-|-|-


| Model (3) : Locale=us |  E  | S |  C | I|
|:----:|:--------:|:-----:|:-----:|:-----:|
|   E  | NaN     | 0.763201 |  0.877799 | 0.849532 |
|   S  | 0.763201     | NaN | 0.830318 |  0.785094 |
|   C  | 0.877799     | 0.830318 | NaN | 0.633192 |
|   I | 0.849532 | 0.785094 | 0.633192 | NaN
|Overall |  nDCG = 0.8882296936440213 | -|-|-|


| Model (3) : Locale=es |  E  | S |  C | I|
|:----:|:--------:|:-----:|:-----:|:-----:|
|   E  | NaN     | 0.819584 |  0.889195 | 0.867129 |
|   S  | 0.819584     | NaN | 0.821450 |  0.777371 |
|   C  | 0.889195     | 0.821450 | NaN | 0.622689 |
|   I | 0.867129 | 0.777371 | 0.622689 | NaN
|Overall |  nDCG = 0.8960547206163951 | -|-|-|



| Model (3) : Locale=jp |  E  | S |  C | I|
|:----:|:--------:|:-----:|:-----:|:-----:|
|   E  | NaN     | 0.838977 |  0.887880 | 0.879604 |
|   S  | 0.819584     | NaN | 0.815444 |  0.793070 |
|   C  | 0.887880     | 0.815444 | NaN | 0.662634 |
|   I | 0.879604 | 0.793070 | 0.662634 | NaN
|Overall |  nDCG = 0.8965548666660679 | -|-|-|




