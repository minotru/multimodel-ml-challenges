"""
Experiment: how data set size N affects accuracy and other metrics?
"""

from functools import partial
from typing import Iterable, Dict, Callable
import logging
import argparse
import json
import time
from collections import Counter

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np
import pandas as pd

from utils import read_data
from utils import preprocess_data
from utils import join_text_columns
from utils import select_popular_classes

logging.basicConfig(level="INFO")
logger = logging.getLogger()

TEXT_COLUMN = "Text"
LABEL_COLUMN = "Area"
MAX_N = 10000
DEFAULT_REPRESENTATION_THRESHOLD = 10

def representation_score(y, K: int) -> float:
    represented_count = sum(k for k in Counter(y).values() if k >= K)
    return represented_count / len(y)

def run_experiment_iteration(
    grid_search: GridSearchCV,
    data_train: pd.DataFrame, 
    data_test: pd.DataFrame
) -> Dict:
    result = dict()

    result["N"] = len(data_train)
    result["num_train_classes"] = data_train[LABEL_COLUMN].nunique()
    result["num_test_classes"] = data_test[LABEL_COLUMN].nunique()

    result["mean_num_samples_per_class"] = data_train[LABEL_COLUMN].value_counts().mean()
    result["median_num_samples_per_class"] = data_train[LABEL_COLUMN].value_counts().median()

    grid_search.fit(data_train[TEXT_COLUMN], data_train[LABEL_COLUMN])

    for metric_name, scorer in grid_search.scorer_.items():
        for area in ["train", "test"]:
            mean_score = grid_search.cv_results_[f"mean_{area}_{metric_name}"][grid_search.best_index_]
            std_score = grid_search.cv_results_[f"std_{area}_{metric_name}"][grid_search.best_index_]

            area_prefix = "train" if area == "train" else "val" 
            result[f"{area_prefix}_{metric_name}"] = mean_score
            result[f"std_{area_prefix}_{metric_name}"] = std_score

        test_score = scorer(grid_search.best_estimator_, data_test[TEXT_COLUMN], data_test[LABEL_COLUMN])
        result[f"test_{metric_name}"] = test_score

    result.update(grid_search.best_params_)

    return result


def get_grid_search(config: Dict) -> GridSearchCV:
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=200, multi_class="ovr"))
    ])

    param_grid = {
        "vectorizer__max_features": [100, 500, 1000, 2000, 5000, 10000, None],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.001, 0.01, 0.5, 1]
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, 
        scoring=["accuracy", "f1_weighted", "f1_macro"],
        refit="f1_macro",
        return_train_score=True,
        cv=3, 
        n_jobs=-1)

    return grid_search

def run_experiment(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    data_train = preprocess_data(data_train)
    data_test = preprocess_data(data_test)

    for data in [data_train, data_test]:
        data[TEXT_COLUMN] = join_text_columns(data, config["text_columns"])

    Ns = np.arange(1000, MAX_N, 1000)
    if Ns[-1] != MAX_N:
        Ns = np.append(Ns, MAX_N)

    grid_search = get_grid_search(config)

    results = []
    for N in Ns:
        logger.info(f"running for N = {N}")
        t0 = time.time()
        
        data_train_sample = data_train.sample(N)
        data_train_sample, data_test_sample = select_popular_classes(
            data_train_sample, data_test, LABEL_COLUMN, config["min_samples_per_class"])

        result  = run_experiment_iteration(grid_search, data_train_sample, data_test_sample)
        result["min_samples_per_class"] = config["min_samples_per_class"]

        representation_score_threshold = config.get("representation_score_threshold", DEFAULT_REPRESENTATION_THRESHOLD)
        result["representation_score"] = representation_score(
            data_train_sample[LABEL_COLUMN], representation_score_threshold)
        result["representation_score_threshold"] = representation_score_threshold

        logger.info(f"completed in {int(time.time() - t0)} seconds")

        results.append(result)

    results = pd.DataFrame(results)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", help="name of the config file from configs/")

    args = parser.parse_args()

    with open(f"configs/{args.config_name}.json", "r") as config_file:
        config = json.load(config_file)

    data_train = read_data("data/raw/issues_train.tsv")
    data_test = read_data("data/raw/issues_test.tsv")

    logger.info(f"running experiment: {args.config_name}")

    t0 = time.time()

    results = run_experiment(data_train, data_test, config)

    logger.info(f"experiment completed in {int(time.time() - t0)} seconds")

    results.to_csv(f"results/{args.config_name}.csv", index=False)

if __name__ == "__main__":
    main()


