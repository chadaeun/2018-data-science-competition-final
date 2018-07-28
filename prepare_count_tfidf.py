import os
import argparse
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def fit_save_count_tfidf(texts, count_path, tfidf_path):
    """
    Fit and save CountVectorizer and TfidfTransformer
    :param texts: list of texts
    :param count_path: Path of CountVectorizer pickle
    :param tfidf_path: Path of TfidfTransformer pickle
    :return: None
    """
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(texts)
    tfidf_transformer = TfidfTransformer().fit(counts)

    with open(count_path, 'wb') as f:
        pickle.dump(count_vectorizer, f)

    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_transformer, f)

def main(args):
    if not os.path.isdir(args.pickle_dir):
        os.makedirs(args.pickle_dir)

    # prepare data
    df = pd.read_csv(args.train_path)

    titles = df['outtitle'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    reviews = df['outreview'].apply(lambda x: ' '.join(ast.literal_eval(x)))

    title_count_path = os.path.join(args.pickle_dir, 'title_CountVectorizer.pickle')
    title_tfidf_path = os.path.join(args.pickle_dir, 'title_TfidfTransformer.pickle')
    review_count_path = os.path.join(args.pickle_dir, 'review_CountVectorizer.pickle')
    review_tfidf_path = os.path.join(args.pickle_dir, 'review_TfidfTransformer.pickle')

    fit_save_count_tfidf(titles, title_count_path, title_tfidf_path)
    fit_save_count_tfidf(reviews, review_count_path, review_tfidf_path)

    print('Result:')
    print('title CountVectorizer: {}'.format(title_count_path))
    print('title TfidfTransformer: {}'.format(title_tfidf_path))
    print('review CountVectorizer: {}'.format(review_count_path))
    print('review TfidfTransformer: {}'.format(review_tfidf_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path of train dataset CSV file')
    parser.add_argument('--pickle_dir', type=str, required=True,
                        help='Directory of CountVectorizer and TfidfTransformer pickle')
    args = parser.parse_args()

    main(args)