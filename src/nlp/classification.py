"""Parametrizable NLP text classification with custom lexicon dictionaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from .datasets import TextDataset
from .dictionaries import DictionaryFeatureExtractor


@dataclass
class NLPParams:
    classifier: str = "LinearSVC"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = 10000
    stop_words: str | None = "english"
    sublinear_tf: bool = True
    svm_c: float = 1.0
    svm_max_iter: int = 2000
    lr_max_iter: int = 1000
    rf_n_estimators: int = 100
    random_state: int = 42
    use_dictionaries: bool = False


@dataclass
class NLPResult:
    accuracy: float
    train_time_s: float
    predictions: np.ndarray
    report: str
    confusion: np.ndarray
    params: NLPParams
    meta: dict = field(default_factory=dict)


CLASSIFIERS = {
    "Naive Bayes": lambda p: MultinomialNB(),
    "LinearSVC": lambda p: LinearSVC(C=p.svm_c, max_iter=p.svm_max_iter),
    "LogisticRegression": lambda p: LogisticRegression(
        max_iter=p.lr_max_iter, solver="saga", random_state=p.random_state
    ),
    "RandomForest": lambda p: RandomForestClassifier(
        n_estimators=p.rf_n_estimators, n_jobs=-1, random_state=p.random_state
    ),
}


def build_pipeline(params: NLPParams, dictionaries: dict[str, list[str]] | None = None) -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=params.ngram_range,
        max_features=params.max_features,
        stop_words=params.stop_words,
        sublinear_tf=params.sublinear_tf,
    )
    if params.use_dictionaries:
        combined = FeatureUnion(
            [
                ("tfidf", tfidf),
                ("dict", DictionaryFeatureExtractor(dictionaries)),
            ]
        )
        return Pipeline([("vec", combined), ("clf", CLASSIFIERS[params.classifier](params))])
    else:
        return Pipeline([("vec", tfidf), ("clf", CLASSIFIERS[params.classifier](params))])


def evaluate(
    dataset: TextDataset,
    params: NLPParams | None = None,
    dictionaries: dict[str, list[str]] | None = None,
) -> NLPResult:
    params = params or NLPParams()
    pipeline = build_pipeline(params, dictionaries)
    t0 = time.perf_counter()
    pipeline.fit(dataset.train_texts, dataset.train_labels)
    train_time = time.perf_counter() - t0
    pred = pipeline.predict(dataset.test_texts)
    acc = accuracy_score(dataset.test_labels, pred)
    report = classification_report(
        dataset.test_labels, pred, target_names=dataset.target_names, zero_division=0
    )
    cm = confusion_matrix(dataset.test_labels, pred)
    return NLPResult(
        accuracy=acc,
        train_time_s=train_time,
        predictions=pred,
        report=report,
        confusion=cm,
        params=params,
        meta={"dataset": dataset.name, "n_train": dataset.n_train, "n_test": dataset.n_test},
    )


def compare_classifiers(
    dataset: TextDataset,
    base_params: NLPParams | None = None,
    dictionaries: dict[str, list[str]] | None = None,
) -> list[NLPResult]:
    base = base_params or NLPParams()
    results = []
    for name in CLASSIFIERS:
        p = NLPParams(
            classifier=name,
            ngram_range=base.ngram_range,
            max_features=base.max_features,
            stop_words=base.stop_words,
            sublinear_tf=base.sublinear_tf,
            svm_c=base.svm_c,
            random_state=base.random_state,
            use_dictionaries=base.use_dictionaries,
        )
        results.append(evaluate(dataset, p, dictionaries))
    return results


def study_ngram(
    dataset: TextDataset,
    ngrams: list[tuple[int, int]] | None = None,
    base: NLPParams | None = None,
    dictionaries: dict[str, list[str]] | None = None,
) -> list[NLPResult]:
    base = base or NLPParams(classifier="LinearSVC")
    ngrams = ngrams or [(1, 1), (1, 2), (2, 2), (1, 3)]
    out = []
    for ng in ngrams:
        p = NLPParams(
            classifier=base.classifier,
            ngram_range=ng,
            max_features=base.max_features,
            stop_words=base.stop_words,
            use_dictionaries=base.use_dictionaries,
        )
        out.append(evaluate(dataset, p, dictionaries))
    return out


def study_max_features(
    dataset: TextDataset,
    values: list[int | None] | None = None,
    base: NLPParams | None = None,
    dictionaries: dict[str, list[str]] | None = None,
) -> list[NLPResult]:
    base = base or NLPParams(classifier="LinearSVC")
    values = values or [500, 1000, 5000, 10000, None]
    out = []
    for mf in values:
        p = NLPParams(
            classifier=base.classifier,
            ngram_range=base.ngram_range,
            max_features=mf,
            stop_words=base.stop_words,
            use_dictionaries=base.use_dictionaries,
        )
        out.append(evaluate(dataset, p, dictionaries))
    return out
