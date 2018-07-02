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

At this point the output file can be used to be the input of the siamese neural network, each row
in the file contains information about a combination of forecasts.

At this point In order to start the training of the neural network we can type:
```bash
$ python train.py --siamese_size=25  --hidden_size=20 --epochs=2 --batch_size=64 --input=data/ds_wit_combinations.csv
```

The script train.py is parameterized, as can be seen in the above example, anyway it can also be started with default 
hyperparameters with:

```bash
$ python train.py --input=data/ds_wit_combinations.csv
```
It is adviceable to first run the script on a small dataset to inspect the output, and modify it accordingly to
preference.

For example the loss function values could be saved in a separate file or plotted on screen with a plot,
by default all the loss function values will be printed on screen.

