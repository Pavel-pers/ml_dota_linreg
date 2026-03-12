import json
import sqlite3
from typing import List, Optional, Dict, Any, Callable, Tuple

import numpy as np
import optuna

from learn_pipeline import LearnPipeline


class ExperimentStorage:
    def __init__(self, db_path: str = "data/artifacts/experiments.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments
            (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                mean_gini REAL NOT NULL
            )
            """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS optuna_runs
            (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                best_gini   REAL    NOT NULL,
                best_params TEXT    NOT NULL,
                n_trials    INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()

    def evaluate(self, name: str, config: "LearnConfig", df: Any, split_masks: List,
                 sparse_list: Optional[List] = None, verbose: bool = False) -> float:
        evaluated = self._conn.execute(
            "SELECT mean_gini FROM experiments WHERE name = ?", (name,)
        ).fetchone()
        if evaluated:
            if verbose:
                print(f"  [{name}] cached: Gini = {evaluated[0]:.6f}")
            return evaluated[0]

        learning = LearnPipeline(**config.pipeline_args())
        scores = learning.evaluate(df, sparse_list or [], split_masks, verbose=verbose)
        mean = float(np.mean(scores))
        self._conn.execute("INSERT INTO experiments (name, mean_gini) VALUES (?, ?)",
                           (name, round(mean, 8)))
        self._conn.commit()
        if verbose:
            print(f"[{name}] Gini = {mean:.6f}")
        return mean

    def evaluate_results(self):
        return self._conn.execute(
            "SELECT name, mean_gini FROM experiments ORDER BY mean_gini DESC"
        ).fetchall()

    def optimize(self, name: str, objective: Callable, n_trials: int = 50,
                 verbose: bool = True, **study_kwargs) -> Tuple[float, Dict[str, Any]]:
        optimized = self._conn.execute(
            "SELECT best_gini, best_params FROM optuna_runs WHERE name = ?", (name,)
        ).fetchone()
        if optimized:
            best_gini, best_params = optimized[0], json.loads(optimized[1])
            if verbose:
                print(f"[{name}] cached: best Gini = {best_gini:.6f}, params = {best_params}")
            return best_gini, best_params

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize", **study_kwargs)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=2)

        self._conn.execute(
            "INSERT INTO optuna_runs (name, best_gini, best_params, n_trials) VALUES (?, ?, ?, ?)",
            (name, round(study.best_value, 8),
             json.dumps(study.best_params, default=str), n_trials))
        self._conn.commit()
        return study.best_value, study.best_params

    def optuna_results(self):
        """Все optuna-запуски, лучшие сверху."""
        return self._conn.execute(
            "SELECT name, best_gini, best_params, n_trials FROM optuna_runs ORDER BY best_gini DESC"
        ).fetchall()
