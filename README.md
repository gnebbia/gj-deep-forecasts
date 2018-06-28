# Good Judgement Neural Network

The goal of this project is to implement a neural network which given a couple of geopolitical forecasts is able 
to determine which one is better.

Details about the dataset and the idea come from the ACE IARPA project
available at this link:
[IARPA Good Judgement
Project](https://www.dni.gov/index.php/newsroom/press-releases/item/1751-iarpa-announces-publication-of-data-from-the-good-judgment-project
"IARPA Good Judgement Project").

The dataset used by the neural network comes from the Good Judgement Project 
which is freely available at the following link:
[Good Judgement Project Dataset](https://doi.org/10.7910/DVN/BPCDH5
"Good Judgement Project Dataset").

## Setup

It is enough to clone the repository with:
```bash 
$ git clone https://github.com/gnebbia/gj-deep-forecasts
```

It is recommended as usual to create a virtual environment and after its activation 
the dependencies can be installed with:
```bash
$ pip install -r requirements.txt
```

## Usage Example

To generate the question file with topic relevance type:
```bash
$ python gjnn/lda.py
```

To preprocess the dataset of forecast with the question generated file type:
```bash
$ python generate_dataset.py --inputsurv=data/survey_fcasts.yr1.tab --inputquest=data/questions_w_topics.csv --outputf=~/ds.csv
```

We still need a preprocessing step used to generate combinations of couples of predictions from the dataset we 
produced in the last step, this step will output us a dataset which can be fed to the neural network:
```bash
$ python generate_pairs_for_siamese_net.py --input=~/ds.csv --output=~/ds_with_combinations.csv
```

At this point the output file can be used to be the input of the siamese neural network, each couple of rows
in the file compose a combination of forecasts, so the even rows will be fed to a branch of the siamese network
while the odd rows will be fed to the other branch of the siamese network.

