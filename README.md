# 2018 Data Science Competition 예선 데이터 분석 보고서 - 차다은 & 김혜경

## Preprocessing

`data/Womens Clothing E-Commerce Reviews.csv` 데이터셋을 전처리합니다.

```
python preprocessing.py --input_path "data/Womens Clothing E-Commerce Reviews.csv" --output_dir "data/"
```

## Prepare Doc2Vec

Doc2Vec sentence embedding model을 준비합니다.

```
python prepare_doc2vec.py --train_path "data/train.csv" --doc2vec_dir "doc2vec/"
```

## Prepare CountVectorizer and TfidfTransformer

CountVectorizer와 TfidfTransformer를 준비합니다.

```
python prepare_count_tfidf.py --train_path "data/binary/train.csv" --pickle_dir "pickle/"
```

## Train

선호도 판별 모델을 구축합니다.

- binary class:

    ```
    --train_path "data/binary/train.csv" --devtest_path "data/binary/devtest.csv" --vectorize count_tfidf --result_path "result/devtest/count_tfidf_multi.csv" --confusion_matrix_dir "confusion_matrix/devtest/binary/count_tfidf/"
    ```

- multi class:

    ```
    --train_path "data/multi/train.csv" --devtest_path "data/multi/devtest.csv" --vectorize count_tfidf --result_path "result/devtest/count_tfidf_multi.csv" --confusion_matrix_dir "confusion_matrix/devtest/multi/count_tfidf/"
    ```

## Crawl Test Data from Web

웹으로부터 테스트 데이터를 수집합니다.

- man:

    ```
    python crawling.py --url https://www.yelp.com/biz/proper-cloth-new-york-3?start --binary_output_path "binary/data/man.csv" --multi_output_path "multi/data/man.csv"
    ```

- woman:

    ```
    python crawling.py --url https://www.yelp.com/biz/primark-boston?start --binary_output_path "binary/data/woman.csv" --multi_output_path "multi/data/woman.csv"
    ```

- cosmetic:

    ```
    python crawling.py --url https://www.yelp.com/biz/sephora-boston?start --binary_output_path "binary/data/cosmetic.csv" --multi_output_path "multi/data/cosmetic.csv"
    ```

- food:

    ```
    python crawling.py --url https://www.yelp.com/biz/yamasho-san-francisco-2?start --binary_output_path "binary/data/food.csv" --multi_output_path "multi/data/food.csv"
    ```

## Evaluate

선호도 판별 모델의 성능을 평가합니다.

- binary class:

    ```
    python evaluate.py --model_path "model/binary_best.pickle" --result_path "result/test/binary.csv" --confusion_matrix_dir "confusion_matrix/test/binary/" --test_paths "data/binary/test.csv" "data/binary/cosmetic.csv" "data/binary/food.csv" "data/binary/man.csv" "data/binary/woman.csv"
    ```

- multi class:

    ```
    python evaluate.py --model_path "model/multi_best.pickle" --result_path "result/test/multi.csv" --confusion_matrix_dir "confusion_matrix/test/multi/" --test_paths "data/multi/test.csv" "data/multi/cosmetic.csv" "data/multi/food.csv" "data/multi/man.csv" "data/multi/woman.csv"
    ```