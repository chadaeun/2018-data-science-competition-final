import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn_evaluation import plot

from lib import vectorize


def main(args):
    print('Preparing...')

    # Load CountVectorizer and TfidfTransformer
    with open(os.path.join(args.pickle_dir, 'review_CountVectorizer.pickle'), 'rb') as f:
        review_count = pickle.load(f)

    with open(os.path.join(args.pickle_dir, 'review_TfidfTransformer.pickle'), 'rb') as f:
        review_tfidf = pickle.load(f)

    with open(os.path.join(args.pickle_dir, 'title_CountVectorizer.pickle'), 'rb') as f:
        title_count = pickle.load(f)

    with open(os.path.join(args.pickle_dir, 'title_TfidfTransformer.pickle'), 'rb') as f:
        title_tfidf = pickle.load(f)

    # Load model
    with open(args.model_path, 'rb') as f:
        clf = pickle.load(f)

    # binary or not
    binary = len(clf.classes_) == 2

    # Init Result File
    result_dir = os.path.split(args.result_path)[0]
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if not os.path.isfile(args.result_path):
        if binary:
            pd.DataFrame(columns=['Dataset Name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']) \
                .to_csv(args.result_path, index=False)
        else:
            pd.DataFrame(columns=['Dataset Name', 'accuracy', 'precision', 'recall', 'f1']) \
                .to_csv(args.result_path, index=False)

    # Init Confusion Matrix Directory
    if args.confusion_matrix_dir:
        if not os.path.isdir(args.confusion_matrix_dir):
            os.makedirs(args.confusion_matrix_dir)

    # Evaluating
    for test_path in args.test_paths:
        test_name = os.path.splitext(os.path.split(test_path)[1])[0]

        test_df = pd.read_csv(test_path)
        test_X, test_y = vectorize.count_tfidf_make_dataset(test_df, review_count, review_tfidf, title_count, title_tfidf)
        pred = clf.predict(test_X)

        # Save Confusion Matrix Image
        if args.confusion_matrix_dir:
            plot.confusion_matrix(test_y, pred)
            plt.savefig(os.path.join(args.confusion_matrix_dir, '{}.png'.format(test_name)))
            plt.clf()

        # Save Result
        result_df = pd.read_csv(args.result_path)

        if binary:
            result_df.loc[len(result_df)] = {
                'Dataset Name': test_name,
                'accuracy': accuracy_score(test_y, pred),
                'precision': precision_score(test_y, pred),
                'recall': recall_score(test_y, pred),
                'f1': f1_score(test_y, pred),
                'roc_auc': roc_auc_score(test_y, pred),
            }
        else:
            result_df.loc[len(result_df)] = {
                'Dataset Name': test_name,
                'accuracy': accuracy_score(test_y, pred),
                'precision': precision_score(test_y, pred, average='weighted'),
                'recall': recall_score(test_y, pred, average='weighted'),
                'f1': f1_score(test_y, pred, average='weighted'),
                # ROC AUC is not available on multi class
            }

        result_df.to_csv(args.result_path, index=False)
        print('{} Done...'.format(test_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default='model/binary_best.pickle',
                        help='Path of trained model')
    parser.add_argument('--pickle_dir', type=str, required=False, default='pickle/',
                        help='Directory of CountVectorizer and TfidfTransformer pickles')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path of result CSV file')
    parser.add_argument('--confusion_matrix_dir', type=str, required=False,
                        help='Directory of confusion matrix images')
    parser.add_argument('--test_paths', nargs='+', type=str, required=True,
                        help='Paths of test CSV files')

    args = parser.parse_args()
    main(args)