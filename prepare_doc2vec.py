import ast
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import argparse

def main(args):
    if not os.path.isdir(args.doc2vec_dir):
        os.makedirs(args.doc2vec_dir)

    # prepare data
    df = pd.read_csv(args.train_path)

    titles = [TaggedDocument(ast.literal_eval(doc), [i]) for i, doc in enumerate(df['outtitle'])]
    reviews = [TaggedDocument(ast.literal_eval(doc), [i]) for i, doc in enumerate(df['outreview'])]

    # train and save
    title_model = Doc2Vec(titles, size=args.title_dim, window=2, workers=4)
    title_model_path = os.path.join(args.doc2vec_dir, 'd2v_title_{}'.format(args.title_dim))
    title_model.save(title_model_path)

    review_model = Doc2Vec(reviews, size=args.review_dim, window=2, workers=4)
    review_model_path = os.path.join(args.doc2vec_dir, 'd2v_review_{}'.format(args.review_dim))
    review_model.save(review_model_path)

    print('Result:')
    print('title: {}'.format(title_model_path))
    print('review: {}'.format(review_model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path of train dataset CSV file')
    parser.add_argument('--doc2vec_dir', type=str, required=True,
                        help='Directory of Doc2Vec models')
    parser.add_argument('--title_dim', type=int, required=False, default=100,
                        help='Dimension of title vectors')
    parser.add_argument('--review_dim', type=int, required=False, default=300,
                        help='Dimension of review vectors')
    args = parser.parse_args()

    main(args)