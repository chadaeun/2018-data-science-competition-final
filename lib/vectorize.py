import numpy as np


def get_doc2vec_vectors(df, doc2vec_model):
    sents = df.values.reshape(-1, 1)
    get_embedding = lambda x: doc2vec_model.infer_vector(str(x[0]).split())
    return np.apply_along_axis(get_embedding, 1, sents)

def get_count_tfidf_vectors(df, count_vect, tfidf_transformer):
    return tfidf_transformer.transform(count_vect.transform(df)).toarray()

def doc2vec_make_dataset(df, review_model, title_model):
    review_vectors = get_doc2vec_vectors(df.loc[:, 'outreview'], review_model)
    title_vectors = get_doc2vec_vectors(df.loc[:, 'outtitle'], title_model)

    X = np.concatenate((review_vectors, title_vectors), axis=1)
    y = df.loc[:, 'label'].values

    return X, y

def count_tfidf_make_dataset(df, review_count, review_tfidf, title_count, title_tfidf):
    review_vectors = get_count_tfidf_vectors(df.loc[:, 'outreview'], review_count, review_tfidf)
    title_vectors = get_count_tfidf_vectors(df.loc[:, 'outtitle'], title_count, title_tfidf)

    X = np.concatenate((review_vectors, title_vectors), axis=1)
    y = df.loc[:, 'label'].values

    return X, y

