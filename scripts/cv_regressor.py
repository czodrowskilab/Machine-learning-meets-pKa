"""
Copyright © 2019 Marcel Baltruschat

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

Source: https://github.com/czodrowskilab/gafp/tree/master/fp
"""

from collections import defaultdict as ddict
from typing import Dict, Any, List

import numpy as np
from sklearn.model_selection import KFold


class CVRegressor:
    """
    Regressor that predicts based on predictions of k models from k-fold CV.
    Accepts any Scikit-learn-like regressor as base regressor. It trains k models
    by doing k-fold CV and stores the individual models. Predictions
    on new samples are done by calculating mean predictions from all models.

    Parameters
    ----------
    est : Any
        Scikit-learn (-like) regressor object. Must contain .fit() and .predict() methods.
    params : Dict[str, Any]
        Regressor parameters
    n_folds : int
        Number of folds for k-fold
    shuffle : bool
        Shuffling of data for CV
    """
    __slots__ = ('est', 'params', 'models', 'n_folds', 'shuffle', 'cv_scores')

    def __init__(self, est: Any, params: Dict[str, Any], n_folds: int = 5, shuffle: bool = True):
        self.est = est
        self.params = params
        self.models = []
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.cv_scores = ddict(list)

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, scoring_funcs: List = (), random_state: int = None) -> None:
        """
        Build a regressor consisting of k-models.

        Parameters
        ----------
        x_data : numpy.ndarray
            Training data
        y_data : numpy.ndarray
            Target values
        scoring_funcs : list
            List of scoring functions to use for evaluating cross-validation results
        random_state : int
            Integer to use for seeding the k-fold split
        """

        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=random_state)
        kf = kf.split(X=x_data, y=y_data)

        # Fit k models and store them
        for train_index, test_index in kf:
            est_tmp = self.est(**self.params)
            est_tmp.fit(x_data[train_index], y_data[train_index])
            if scoring_funcs:
                test_pred = est_tmp.predict(x_data[test_index])
                for sf in scoring_funcs:
                    self.cv_scores[str(sf).split(' ')[1]].append(sf(y_data[test_index], test_pred))
            self.models.append(est_tmp)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """
        Predict using prediction mean from k models.

        Parameters
        ----------
        x_data : numpy.ndarray
            Samples to predict

        Returns
        -------
        numpy.ndarray
            Predicted values
        """

        return np.mean([m.predict(x_data) for m in self.models], axis=0)
