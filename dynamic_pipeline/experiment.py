"""
How the optimal pipeline depends on training data set size?

Given: Dtrain[:N], Dtest
Where D: (Title, Class)

Optimal pipeline - with the highest metric(Dtest)

Pseudo-code of the experiment:
metrics = []
for N in Ns:
    for pipeline in pipelines:
        pipeline.fit(Dtrain[:N])
        metric_value = metric(pipeline, Dtest)
        metrics.append({
            'pipeline_name': pipeline.name,
            'N': N,
            'metric': metric_value
        })
        
Considered pipeline components:
1. Vectorization: 
    * CountVectorizer vs TfidfVectorizer?
    * max_features? 
    * min_df vs max_df?
2. Feature selection:
    * Use or do not use?
    * Algorithm: 
2. Dim. reduction - TruncatedSVD:
    * Use or do not use?
    * n_components?
3. Model:
    * LogisticRegression
    * SGDClassifier:
        * loss?
        * l1, l2 rate?
        * alpha?
    * MultinomialNB 
    * ComplementNB

Important questions to answer:
1. max_features/N ratio impact
2. TruncatedSVD: when to use, N vs M vs n_components
3. TruncatedSVD vs model
"""

from typing import List, Dict, Iterable
import logging

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import clone

logging.basicConfig(level="INFO")
logger = logging.getLogger()

def run_experiment(
    X_train, y_train,
    X_test, y_test,
    Ns: List[int],
    pipeline: Pipeline,
    params_list: Iterable[Dict],
    metric,
) -> pd.DataFrame:
    logging.info(f"len(Ns) = {len(Ns)}, len(params_list) = {len(params_list)}, {len(Ns) * len(params_list)} combinations to check in total")

    results = []
    for N_idx, N in enumerate(Ns):
        logger.info(f"running N = {N}, {N_idx + 1}'s' out of {len(Ns)}")
        for params_idx, params in enumerate(params_list):
            logger.info(f"running {params_idx}'s params out of {len(params_list)}")

            pipeline = clone(pipeline)
            pipeline.set_params(**params)

            pipeline.fit(X_train[:N], y_train[:N])
            score = metric(y_test, pipeline.predict(X_test))
            results.append({
                "N": N,
                "M": pipeline.steps[-1][1].coef_.shape[1],
                "params": params,
                "score": score,
            })
    
    return pd.DataFrame(results)

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import ParameterGrid
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import accuracy_score

    TEXT_COLUMN = "Title"
    LABEL_COLUMN = "Area"

    data_train = read_data("data/raw/issues_train.tsv")
    data_test = read_data("data/raw/issues_test.tsv")

    X_train, y_train = data_train[TEXT_COLUMN], data_train[LABEL_COLUMN]
    X_test, y_test = data_test[TEXT_COLUMN], data_test[LABEL_COLUMN]

    pipeline = Pipeline([
        ("vec", None),
        ("svd", None),
        ("model", None)
    ])

    grid = ParameterGrid({
        "vec": [TfidfVectorizer()],
        "vec__max_features": [500, 1000, 5000],
        "svd": [TruncatedSVD()],
        "svd__n_components": [50, 100, 200, 300],
        "model": [LogisticRegression()]
    })

    Ns = [500, 1000, 1500, 2000, 2500, 5000, 10000]

    results = run_experiment(
        X_train, y_train,
        X_test, y_test,
        Ns,
        pipeline,
        grid,
        accuracy_score
    )

    results = results.sort_values("score", ascending=False)
    results.to_csv("results/result.csv", index=False)


if __name__ == "__main__":
    main()
