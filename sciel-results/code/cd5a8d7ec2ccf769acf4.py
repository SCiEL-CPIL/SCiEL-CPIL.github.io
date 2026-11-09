"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    # Least-squares fit of a degree-4 polynomial feature expansion.
    d = X_train.shape[1]
    max_degree = 4
    
    def features(X):
        cols = [np.ones(len(X))]
        for degree in range(1, max_degree + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        return np.column_stack(cols)
    
    F_train = features(X_train)
    w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
    
    def predict(X):
        return features(X) @ w
    
    return predict
# EVOLVE-BLOCK-END
