# import tensorflow as tf
import pandas as pd
from IPython.display import display

# Setup plotting
import matplotlib.pyplot as plt


if __name__ == '__main__':
    shoes = pd.read_csv('StockX-Data-Contest-2019.csv')

    # Create training and validation splits
    df_train = shoes.sample(frac=0.7, random_state=0)  # frac is the fraction of the data set to use
    df_valid = shoes.drop(df_train.index)  # drops (slices) the training data from shoes using the index (shoes - training = valid)
    print(df_train.head(4))

    # Scale to [0, 1]
    max_ = df_train.max(axis=0)
    min_ = df_train.min(axis=0)
    df_train = (df_train - min_) / (max_ - min_)
    df_valid = (df_valid - min_) / (max_ - min_)

    # Split features and target
    X_train = df_train.drop('Sale Price', axis=1)
    X_valid = df_valid.drop('Sale Price', axis=1)
    y_train = df_train['Sale Price']
    y_valid = df_valid['Sale Price']
    print(X_train.head(4))
    print(X_valid.head(4))
    print(y_train.head(4))
    print(y_valid.head(4))

    print(shoes.head())