The code considers a python installation of python 3.7.11 and various packages with versions suggested in requirement files.

To run the code inside the repository we provide a requirements file.

To run neural networks the use of tensorflow-gpu=2.8.2 instead of tensorflow=2.8.2 is suggested.

The requirement files uses a more general cpu installation but the neural networks training would require too much time on cpu, so a change to a gpu or the use of freely available notebook environmnets with gpus is suggested for the NN notebooks.

We can first create an environment with conda:
```
    conda create --name "submission_environment" python=3.7.11
```
We activate the created environment:
```
    conda activate submission_environment
```
We then use "requirements.txt" to install the required packages using pip
```
    pip install -r requirements.txt
```
The code is organized in various notebooks.

The code is organized mainly in 3 parts.

Inside each part we numbered the notebooks to make it easier to replicate our results

# Placing the dataset

The files provided in the challenge should be placed inside the ./dataset/original_data folder to be correctly used in the successive steps.

# Data preparation

In the ./Data_preparation folder there are the notebooks used to prepare the dataset used by the various models.

## Clear_ids 

This notebook changes the IDs used to represent both sessions and items using contiguous numbers.
In this notebook we generate modified versions of the originally provided dataset and we also create the URM( User Ratings matrix) in a scipy sparse format that will be used by different recommenders.

## ICM_creation

This notebook generates the ICM ( Item Content Matrix) in a scipy sparse format. 
The general idea is to map each couple of (category_id,feature_id) into a new id and after performing this mapping we also consider category_id alone as a feature. This last choice is to laverage the information provided in the dataset explanation ( third dot of Content data ).

>Some items may not share any feature_category_ids if they are different types of items, for example trousers will share very few category ids with shirts and will share even less category ids with sandals.

This explains that even if the value of a category is different for two items if they share the same category id they can be partially similar.

## Auto_encoder_features

This notebook uses tries to learn an autoencoder over the ICM to reduce the dimensionality of the features.
This is done to allow feeding the features information to different kind of neural networks as pretrained weights of embedding layers.

## Prepare_dataset_NN

This notebook creates the dataset used by the neural networks. The items seen in a session are grouped together and we compute different features of a session like the lenght in seconds, the similarity between the items seen in the session and other summary statistics of the sessions.

# Models for ensemble

In the ./Models_for_ensemble folder there are the notebooks used to generate candidates for different sessions. In particular we consider two kinds of models in this folder.
1) Traditional Recommender Systems, in particular Collaborative Filtering, Content Based Filtering and Graph Based Recommenders.
2) NN Recommenders.

In particular we split the train dataset provided leaving out the last month of interactions (our validation set), this is done to emulate the splitting done to generate the test set provided in the challenge.
We don't take special actions to account the fact that test sessions are randomly split between 50% and 100% of their actual length.
In these notebooks we generate candidate item for both the validation and test sets.

## Traditional_Recommenders

This notebook performs training of different kind of traditional recommenders:

1) User Collaborative Filtering Recommenders: To avoid temporal leakage we consider for each validation session the Top K closest sessions among the train sessions. So we don't consider similarities between validation instances and other validation sessions.

2) Item Collaborative Filtering Recommenders: We compute similarities between items starting from the URM that considers only the train sessions leaving out the validation sessions. After computing the similarities between the items, the similarity matrix is used to predict the expected score of the various item for the validation sessions.

3) Item Content Based Filtering Recommenders: We compute  similarities between items starting from the ICM.
After computing the similarities between the items, the similarity matrix is used to predict the expected score of the various item for the validation sessions.

4) Graph Based Recommenders: We compute similarities between items starting from the URM that considers only the train sessions leaving out the validation sessions. After computing the similarities between the items, the similarity matrix is used to predict the expected score of the various item for the validation sessions.

For the first 3 classes of Recommenders we train 8 different models using different distance metrics:
cosine, pearson, jaccard, tanimoto, asymmetric cosine, adjusted cosine, dice and tversky. 

For each session among the validation sessions we compute 100 candidates for each recommender considered.
Combining these recommenders we are able to find the correct item among the candidates for around 70% of the sessions. The main problem is that for a lot of users that item is not in the top 100  considering the scores resulting from the different recommenders.

## Traditional_Recommenders-leaderboard & Traditional_Recommenders-final

The same models are trained again this time without leaving out the validation month.
The same considerations to avoid leakage from the validation set are also replicated here for the test set.

The only change is that we compute 150 candidates for each recommender instead of 100.

## Transformer (a-b)

These notebooks contains the code to train a transformer based NN.
In the (a) notebook we train the network in a supervised fashion over all the train sessions except the last month, for which we compute the top 100 condidates.
In the (b) notebook we train the network in a supervised fashion over all the train sessions and compute the top 100 candidates for the test sessions.

## LSTM (a-b)

These notebooks contains the code to train a Bidirectional LSTM based NN.
In the (a) notebook we train the network in a supervised fashion over all the train sessions except the last month, for which we compute the top 100 condidates.
In the (b) notebook we train the network in a supervised fashion over all the train sessions and compute the top 100 candidates for the test sessions.

## GRU (a-b)

These notebooks contains the code to train a GRU based NN.
In the (a) notebook we train the network in a supervised fashion over all the train sessions except the last month, for which we compute the top 100 condidates.
In the (b) notebook we train the network in a supervised fashion over all the train sessions and compute the top 100 candidates for the test sessions.

# Final submission

In the ./Final_submission folder there are the notebooks used to train a set of 10 LGBM Rankers on 10-folds of the list of candidates generated before.

## dressipi-lgbmranker-training_k

In this notebook we perform the training of 10 LGBM Rankers.
We first merge the suggestions of the various recommenders, we compute a dataset composed of ```(session, candidate, session_features and scores given by the various recommenders to the specific couple of session and candidate)```.
We extract only the session for which there is the correct item among the list of candidates and drop all other sessions, we kept around 76.5% of sesssions.
We then get the list of sessions and perform k-fold(k=10) on this list.
We get 10 sets of train and validation sessions and we train 10 LGBM rankers over these sets.
We used MAP@100 as a metric which in presence of only one target item is equivalent to the evaluation metric MRR@100.
The result obtained is around 0.257 MRR@100, if we consider the fact that this is the score over 76% of the sessions and the reamining 24% has a score of 0 since the correct items is not in the candidates, we get a result of 0.257*0.765 which is around 0.1965. 

## dressipi-lgbmranker-(leaderboard|final)_k

In this notebook the 10 models learned in the previous notebook are used to predict the score for each candidate for the (leaderboard|final) test sets.
We used the mean to combine the scores of the 10 models.
The use of models trained on different splits of the candidates dataset we created and the combination of their predictions allows to have stable results and a higher correlation between the scores obtained during training on the last month of the train sessions and the scores obtained on the sessions in the test month. 
