from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
import argparse
import os

from lib import utils


def main(args):
    baseurl = args.url
    binary_output_path = args.binary_output_path
    multi_output_path = args.multi_output_path

    # Set output directory
    binary_output_dir = os.path.split(binary_output_path)[0]
    if not os.path.isdir(binary_output_dir):
        os.makedirs(binary_output_dir)

    multi_output_dir = os.path.split(multi_output_path)[0]
    if not os.path.isdir(multi_output_dir):
        os.makedirs(multi_output_dir)

    n = 20
    ratinglist = []
    reviewlist = []
    titlelist = []

    while n < 1000:
        # Get HTML document
        urladress = baseurl + '=' + str(n)
        html = urlopen(urladress)
        bsobj = BeautifulSoup(html, 'html.parser')

        for i in bsobj.findAll('div', {'class': 'review-content'}):
            # Extract rating and review
            rating = i.find('div', {'class': 'i-stars'})
            review = i.find('p', {'lang': 'en'})

            p = re.compile('title="(.+) star rating"')
            rating = int(float(p.search(str(i)).group(1)))
            review = re.sub('<.*?>', '', str(review))

            # Preprocess review and title
            # We will use first sentence of reivew as title
            title = utils.text_preprocess(sent_tokenize(review)[0])
            review = utils.text_preprocess(' '.join(sent_tokenize(review)[1:]))

            ratinglist.append(rating)
            reviewlist.append(review)
            titlelist.append(title)
        n += 20

    df = pd.DataFrame({'outtitle': titlelist, 'outreview': reviewlist, 'rating': ratinglist})

    # binary test dataset
    binary_df = df[df.rating != 3].apply(lambda x: utils.label_binary(x, 'rating'), axis=1)
    binary_df.to_csv(binary_output_path ,index=False)

    # multi test dataset
    multi_df = df.apply(lambda x: utils.label_multi(x, 'rating'), axis=1)
    multi_df.to_csv(multi_output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, type=str,
                        help='Target URL')
    parser.add_argument('--binary_output_path', required=True, type=str,
                        help='Path of binary class labeled output CSV file')
    parser.add_argument('--multi_output_path', required=True, type=str,
                        help='Path of multi class labeled output CSV file')

    args = parser.parse_args()
    main(args)