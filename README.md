## ASTE-CODE
This is code for Exploiting Duality in Aspect Sentiment Triplet Extraction with Language Prompts.

### Environment
Python 3.7.11
Requirements are list in requirements.txt
We run our code in Ubuntu 20.04.2 LTS, Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, GeForce RTX 3090 24 G.


### Dataset

We use the dataset:  [ASTE-DATA-V2](https://github.com/xuuuluuu/SemEval-Triplet-data)

### Run

Running script is train.sh

there are some arguments need to be filled:

--save_model_path: the model weith will be save in this file in model folder

--data_set: the dataset name, e.g. 14res,15res,14lap,16res

--data_dir: the path of all data_set.  For example, path of "14res" is "/data/astev2/14res", data_dir will be set as "/data/astev2/"
