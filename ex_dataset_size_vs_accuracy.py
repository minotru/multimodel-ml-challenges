"""
Experiment: how data set size N affects accuracy and other metrics?
"""

from functools import partial
from typing import Iterable, Dict, Callable
import logging

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

logging.basicConfig(level="INFO")
logger = logging.getLogger()

TEXT_COLUMN = "Text"
TEXT_COLUMNS = ["Title", "Description"]
LABEL_COLUMN = "Area"
MAX_N = 10000
MIN_SAMPLES_PER_CLASS = 10

def experiment_iteration(
    grid_search: GridSearchCV,
    data_train: pd.DataFrame, 
    data_test: pd.DataFrame,
) -> Dict:
    result = dict()

    result["N"] = len(data_train)
    result["K_train_classes"] = data_train[LABEL_COLUMN].nunique()

    grid_search.fit(data_train[TEXT_COLUMN], data_train[LABEL_COLUMN])

    for metric_name, scorer in grid_search.scorer_.items():
        train_score = grid_search.cv_results_[f"mean_train_{metric_name}"][grid_search.best_index_]
        val_score = grid_search.cv_results_[f"mean_test_{metric_name}"][grid_search.best_index_]
        test_score = scorer(grid_search.best_estimator_, data_test[TEXT_COLUMN], data_test[LABEL_COLUMN])

        result[f"train_{metric_name}"] = train_score
        result[f"val_{metric_name}"] = val_score
        result[f"test_{metric_name}"] = test_score

    result.update(grid_search.best_params_)

    return result


def main():
    data_train = read_data("data/raw/issues_train.tsv")
    data_test = read_data("data/raw/issues_test.tsv")

    data_train = preprocess_data(data_train)
    data_test = preprocess_data(data_test)

    for data in [data_train, data_test]:
        data[TEXT_COLUMN] = join_text_columns(data, TEXT_COLUMNS)

    Ns = np.arange(1000, MAX_N, 1000)
    if Ns[-1] != MAX_N:
        Ns = np.append(Ns, MAX_N)

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=200, multi_class="ovr"))
    ])

    param_grid = {
        "vectorizer__max_features": [100, 500, 1000, 2000, 5000, 10000, None],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.001, 0.01, 0.5, 1]
    }

    # metrics = {
    #     "accuracy": accuracy_score,
    #     "f1_weighted": partial(f1_score, average="weighted")
    # }

    grid_search = GridSearchCV(
        pipeline, param_grid, 
        scoring=["accuracy", "f1_weighted"],
        refit="f1_weighted",
        return_train_score=True,
        cv=3, 
        n_jobs=-1)

    results = []
    for N in Ns:
        logger.info(f"running for N = {N}")
        result  = experiment_iteration(grid_search, data_train[:N], data_test)
        results.append(result)

    results = pd.DataFrame(results)
    results.to_csv("results/dataset_size_vs_accuracy_join_texts.csv", index=False)

if __name__ == "__main__":
    main()


