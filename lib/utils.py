import re

STOP_WORDS = {'a',
     'about',
     'above',
     'after',
     'again',
     'against',
     'ain',
     'all',
     'am',
     'an',
     'and',
     'any',
     'are',
     'aren',
     "aren't",
     'as',
     'at',
     'be',
     'because',
     'been',
     'before',
     'being',
     'below',
     'between',
     'both',
     'by',
     'can',
     'couldn',
     "couldn't",
     'd',
     'did',
     'didn',
     "didn't",
     'do',
     'does',
     'doesn',
     "doesn't",
     'doing',
     'don',
     "don't",
     'down',
     'during',
     'each',
     'few',
     'for',
     'from',
     'further',
     'had',
     'hadn',
     "hadn't",
     'has',
     'hasn',
     "hasn't",
     'have',
     'haven',
     "haven't",
     'having',
     'he',
     'her',
     'here',
     'hers',
     'herself',
     'him',
     'himself',
     'his',
     'how',
     'i',
     'if',
     'in',
     'into',
     'is',
     'isn',
     "isn't",
     'it',
     "it's",
     'its',
     'itself',
     'just',
     'll',
     'm',
     'ma',
     'me',
     'mightn',
     "mightn't",
     'mustn',
     "mustn't",
     'my',
     'myself',
     'needn',
     "needn't",
     'nor',
     'now',
     'o',
     'of',
     'on',
     'once',
     'only',
     'or',
     'other',
     'our',
     'ours',
     'ourselves',
     'out',
     'over',
     'own',
     're',
     's',
     'same',
     'shan',
     "shan't",
     'she',
     "she's",
     'should',
     "should've",
     'shouldn',
     "shouldn't",
     'so',
     'some',
     'such',
     't',
     'than',
     'that',
     "that'll",
     'the',
     'their',
     'theirs',
     'them',
     'themselves',
     'then',
     'there',
     'these',
     'they',
     'this',
     'those',
     'through',
     'to',
     'under',
     'until',
     'up',
     've',
     'very',
     'was',
     'wasn',
     "wasn't",
     'we',
     'were',
     'weren',
     "weren't",
     'what',
     'when',
     'where',
     'which',
     'while',
     'who',
     'whom',
     'why',
     'will',
     'with',
     'won',
     "won't",
     'wouldn',
     "wouldn't",
     'y',
     'you',
     "you'd",
     "you'll",
     "you're",
     "you've",
     'your',
     'yours',
     'yourself',
     'yourselves'
}

def text_preprocess(text):
    """
    Preprocess text
    :param text: text
    :return: preprocessed text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = [w for w in text.split() if w not in STOP_WORDS]

    return text

def label_binary(row, key):
    """
    Label binary class to row
    x <= 3 : 0
    x > 3 : 1
    :param row: pandas Series
    :return: binary class labeled row
    """
    row['label'] = 1 if row[key] > 3 else 0
    del row[key]
    return row

def label_multi(row, key):
    """
    Label multi class to row
    x < 3 : 0
    x == 3 : 1
    x > 3 : 2
    :param row: pandas Series
    :return: multi class labeled row
    """
    if row[key] < 3:
        row['label'] = 0
    elif row[key] == 3:
        row['label'] = 1
    else:
        row['label'] = 2

    del row[key]

    return row