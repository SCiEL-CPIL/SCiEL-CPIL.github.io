import numpy as np


def heuristics(prizes, weights):
        # Ensure prizes is 1D
        if prizes.ndim == 2:
            p = prizes.sum(axis=0)
        else:
            p = prizes
        
        # Ensure weights is 2D (K, N)
        # weights shape (5, 100)
        w = weights
        
        # Calculate sum of weights per item
        w_sum = w.sum(axis=0)
        
        # Calculate competition pressure
        # p shape (100,)
        # w shape (5, 100)
        # overlap(i, j) = dot(w[:, i], w[:, j])
        # We want matrix O (100, 100) where O[i, j] = dot(w[:, i], w[:, j])
        # This is w.T @ w
        overlap_matrix = w.T @ w  # Shape (100, 100)
        
        # Zero out diagonal (self-competition is not "other items")
        np.fill_diagonal(overlap_matrix, 0)
        
        # Pressure[i] = sum_j p[j] * overlap[i, j]
        # p is (100,), overlap is (100, 100)
        # pressure = overlap_matrix @ p  # Shape (100,)
        pressure = overlap_matrix @ p
        
        # Heuristic
        # Avoid division by zero
        denom = w_sum + pressure + 1e-9
        h = p / denom
        
        # Ensure positive
        h = np.maximum(h, 1e-9)
        
        return h
