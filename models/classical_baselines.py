from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class BaselineSpec:
    name: str
    estimator: Any
    description: str


def build_logistic_regression(seed: int = 42) -> BaselineSpec:
    # Simple linear baseline with probability output.
    est = LogisticRegression(
        random_state=seed,
        solver="liblinear",
        max_iter=300,
    )
    return BaselineSpec(
        name="logreg",
        estimator=est,
        description="Logistic Regression (linear baseline, liblinear, max_iter=300)",
    )


def build_small_mlp(seed: int = 42) -> BaselineSpec:
    # Compact neural baseline (input -> 64 -> 32 -> output).
    est = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=256,
        max_iter=100,
        random_state=seed,
    )
    return BaselineSpec(
        name="mlp",
        estimator=est,
        description="Small MLP (64,32), Adam, max_iter=100",
    )


def build_decision_tree(seed: int = 42) -> BaselineSpec:
    # Lightweight nonlinear baseline with moderate regularization.
    est = DecisionTreeClassifier(
        random_state=seed,
        max_depth=8,
        min_samples_leaf=10,
    )
    return BaselineSpec(
        name="tree",
        estimator=est,
        description="Decision Tree (max_depth=8, min_samples_leaf=10)",
    )


def get_all_baselines(seed: int = 42) -> list[BaselineSpec]:
    return [
        build_logistic_regression(seed=seed),
        build_small_mlp(seed=seed),
        build_decision_tree(seed=seed),
    ]
