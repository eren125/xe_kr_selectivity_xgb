# xe_kr_selectivity_xgb
Scripts to train an ML model for Xe/Kr selectivity prediction based on energy descriptors

## Data

The raw data is available in the *raspa/*, *graed/* and *zeo/* directories.

The cleaned data is available in *data/*, where the data is already split into test and train sets using the script *0_data_clean.py*.

## Hyperparameter search

The model's hyperparameters are defined using the script *1_model_xgb_HPsearch.py*.

## Training and testing

The model is trained on the training set *data/train.csv* using the hyperparameters previously provided and printed in *data/hyperparameters.txt*.
Then tested on *data/test.csv*.

Further testings are done in *3_test.py*, with plots comparing the performance of the ML model.

## Interpretation

SHAP library is used to interpret the ML model in *4_shap_interpret.py*. We can find feature importance plots, dependence plots and local interpretations on examples taken from the test set.

## Descriptors

Some visualizations of the relationships between descriptors can be found in *5_feature_viz.py*

## Feature exploration

*99_feature_test_cv.py* allows exploring different combinations of features to find the best combination. The different models are tested using a 5-fold cross validation on the training set *data/train_all.csv*.

## Paper

A preprint version of this work can be found at:

https://chemrxiv.org/engage/chemrxiv/article-details/64ba6061b053dad33aa3e1c8