import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn_evaluation import plot
from gensim.models import Doc2Vec

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from lib import vectorize
"""관련 임포트
from imblearn.combine import SMOTEENN
"""



def main(args):
    print('Preparing...')

    # Load Dataset
    train_df = pd.read_csv(args.train_path)
    devtest_df = pd.read_csv(args.devtest_path)

    if args.vectorize == 'doc2vec':
        title_doc2vec = Doc2Vec.load(args.doc2vec_title_path)
        review_doc2vec = Doc2Vec.load(args.doc2vec_review_path)

        train_X, train_y = vectorize.doc2vec_make_dataset(train_df, review_doc2vec, title_doc2vec)
        devtest_X, devtest_y = vectorize.doc2vec_make_dataset(devtest_df, review_doc2vec, title_doc2vec)

    elif args.vectorize == 'count_tfidf':
        with open(os.path.join(args.pickle_dir, 'review_CountVectorizer.pickle'), 'rb') as f:
            review_count = pickle.load(f)

        with open(os.path.join(args.pickle_dir, 'review_TfidfTransformer.pickle'), 'rb') as f:
            review_tfidf = pickle.load(f)

        train_X, train_y = vectorize.count_tfidf_make_dataset(train_df, review_count, review_tfidf)
        devtest_X, devtest_y = vectorize.count_tfidf_make_dataset(devtest_df, review_count, review_tfidf)
        
        """ 언더&오버샘플적용.
        smote_enn=SMOTEENN(random_state=0)
        train_X_, train_y = smote_enn.fit_sample(train_X,train_y)
        devtest_X, devtest_y = smote_enn.fit_sample(devtest_X,devtest_y)
        """


    else:
        raise ValueError('vectorize method must be doc2vec or count_tfidf')

    # binary or not
    binary = len(np.unique(train_y)) == 2

    # Init Result File
    result_dir = os.path.split(args.result_path)[0]
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if not os.path.isfile(args.result_path):
        if binary:
            pd.DataFrame(columns=['Model Name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])\
                .to_csv(args.result_path, index=False)
        else:
            pd.DataFrame(columns=['Model Name', 'accuracy', 'precision', 'recall', 'f1'])\
                .to_csv(args.result_path, index=False)

    # Init Confusion Matrix Directory
    if args.confusion_matrix_dir:
        if not os.path.isdir(args.confusion_matrix_dir):
            os.makedirs(args.confusion_matrix_dir)

    # Model list
    clf_models = [
        DecisionTreeClassifier,
        LogisticRegression,
        Perceptron,
        RandomForestClassifier,
        LinearSVC,
    ]

    # Training
    for model in clf_models:
        clf = model(class_weight='balanced')
        clf.fit(train_X, train_y)
        """
        pred = clf.predict_proba(devtest_X)
        proba=pd.DataFrame(clf.predict_proba(devtest_X))
        pred=proba.applymap(lambda x: 1 if x>0.7 else 0).values
        
        """
        # Save Best Model
        if model == LogisticRegression:
            if binary:
                best_path = 'model/binary_best.pickle'
            else:
                best_path = 'model/multi_best.pickle'

            with open(best_path, 'wb') as f:
                pickle.dump(clf, f)

        # Save Confusion Matrix Image
        if args.confusion_matrix_dir:
            plot.confusion_matrix(devtest_y, pred)
            plt.savefig(os.path.join(args.confusion_matrix_dir, '{}.png'.format(model.__name__)))
            plt.clf()

        # Save Result
        result_df = pd.read_csv(args.result_path)

        if binary:
            result_df.loc[len(result_df)] = {
                'Model Name': model.__name__,
                'accuracy': accuracy_score(devtest_y, pred),
                'precision': precision_score(devtest_y, pred),
                'recall': recall_score(devtest_y, pred),
                'f1': f1_score(devtest_y, pred),
                'roc_auc': roc_auc_score(devtest_y, pred),
            }
        else:
            result_df.loc[len(result_df)] = {
                'Model Name': model.__name__,
                'accuracy': accuracy_score(devtest_y, pred),
                'precision': precision_score(devtest_y, pred, average='weighted'),
                'recall': recall_score(devtest_y, pred, average='weighted'),
                'f1': f1_score(devtest_y, pred, average='weighted'),
                # ROC AUC is not available on multi class
            }

        result_df.to_csv(args.result_path, index=False)
        print('{} Done...'.format(model.__name__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=False, default='data/binary/train.csv',
                        help='Path of train dataset')
    parser.add_argument('--devtest_path', type=str, required=False, default='data/binary/devtest.csv',
                        help='Path of devtest dataset')
    parser.add_argument('--vectorize', required=True, type=str,
                        help='Vectorize method (doc2vec or count_tfidf)')
    parser.add_argument('--doc2vec_title_path', required=False, type=str, default='doc2vec/d2v_title_100',
                        help='Path of title Doc2Vec model (ignored when vectorize is not doc2vec)')
    parser.add_argument('--doc2vec_review_path', required=False, type=str, default='doc2vec/d2v_review_300',
                        help='Path of review Doc2Vec model (ignored when vectorize is not doc2vec)')
    parser.add_argument('--pickle_dir', type=str, required=False, default='pickle/',
                        help='Directory of CountVectorizer and TfidfTransformer pickles (ignored when vectorize is not count_tfidf')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path of result CSV file')
    parser.add_argument('--confusion_matrix_dir', type=str, required=False,
                        help='Directory of confusion matrix images')

    args = parser.parse_args()
    main(args)
