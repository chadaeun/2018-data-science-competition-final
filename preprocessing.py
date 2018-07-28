import pandas as pd
import argparse
import os
from sklearn.utils import shuffle

from lib import utils


def append_preprocessed_text(row):
    """
    Append preprocessed title and review to row
    :param row: Pandas Series from our dataset 'Womens Clothing E-commerse Reviews.csv'
    :return: row with preprocessed title and review
    """
    row['outtitle'] = utils.text_preprocess(row['Title'])
    row['outreview'] = utils.text_preprocess(row['Review Text'])

    return row

def divide_save_dataset(df, output_dir):
    """
    Divide and save dataset into train(80%), devtest(10%), test(10%)
    :param df: Pandas Dataframe
    :param output_dir: directory of outputs
    :return: None
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # divide dataset into train, test, devtest
    df = shuffle(df)
    idx_train = int(len(df) * 0.8)
    idx_devtest = int(len(df) * 0.9)

    train_path = os.path.join(output_dir, 'train.csv')
    devtest_path = os.path.join(output_dir, 'devtest.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train = df.iloc[:idx_train]
    train.to_csv(train_path, index=False)

    devtest = df.iloc[idx_train:idx_devtest]
    devtest.to_csv(devtest_path, index=False)

    test = df.iloc[idx_devtest:]
    test.to_csv(test_path, index=False)

def main(args):

    df = pd.read_csv(args.input_path, index_col=0)
    df = df.filter(['Title', 'Review Text', 'Rating'], axis=1)

    # We will use only title, review text, and rating columns
    df = df.dropna(axis=0, how='any')
    df = df.apply(append_preprocessed_text, axis=1)

    df = df.filter(['outtitle', 'outreview', 'Rating'], axis=1)

    binary_df = df[df.Rating != 3].apply(lambda x: utils.label_binary(x, 'Rating'), axis=1)
    divide_save_dataset(binary_df, os.path.join(args.output_dir, 'binary'))

    multi_df = df.apply(lambda x: utils.label_multi(x, 'Rating'), axis=1)
    divide_save_dataset(multi_df, os.path.join(args.output_dir, 'multi'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', required=True, type=str,
                        help='Path of input data CSV file')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Path of output preprocessed data CSV files (train, devtest, test)')

    args = parser.parse_args()
    main(args)