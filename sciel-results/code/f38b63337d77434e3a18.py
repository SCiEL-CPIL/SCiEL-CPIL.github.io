# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free optimizer using adaptive coordinate descent with
    dynamic rotation of the search basis via principal directions of
    recent evaluation history. Focuses on top-2 principal components
    to align with ridge and normal directions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initialize at a random point in the box
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        calls = 1

        # Maintain a history of (direction, improvement) pairs for PCA
        history_dirs = []  # list of unit vectors of successful moves
        history_imps = []  # magnitudes of improvement

        # Initial basis: identity
        basis = np.eye(d)

        # Step size for coordinate descent
        step = 0.5

        # PCA update interval
        pca_interval = 15
        # Minimum number of history entries before rotating
        min_hist = 8
        # Number of recent entries to use for PCA
        pca_window = 10

        for i in range(B - 1):
            # Determine search directions: current basis vectors
            # For each axis in the rotated frame, try a line search
            for k in range(d):
                if calls >= B:
                    break
                direction = basis[:, k]
                # Adaptive line search: try several step sizes
                best_x = x
                best_f = fx
                # Try positive and negative directions with more step sizes
                for sign in [1, -1]:
                    for mult in [1.0, 0.7, 0.5, 0.3, 0.2]:
                        if calls >= B:
                            break
                        trial = x + sign * step * mult * direction
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls += 1
                        if ft < best_f:
                            best_f = ft
                            best_x = trial

                if best_f < fx:
                    # Record the move
                    move = best_x - x
                    norm = np.linalg.norm(move)
                    if norm > 1e-12:
                        history_dirs.append(move / norm)
                        history_imps.append(fx - best_f)
                    x = best_x
                    fx = best_f
                    # Increase step size on success
                    step = min(step * 1.2, 3.0)
                else:
                    # No improvement; shrink step
                    step = max(step * 0.65, 0.005)

            # Periodically update the basis using PCA of recent history
            if len(history_dirs) >= min_hist and (i + 1) % pca_interval == 0:
                # Take the most recent history entries
                n_use = min(pca_window, len(history_dirs))
                recent_dirs = np.array(history_dirs[-n_use:])
                recent_imps = np.array(history_imps[-n_use:])
                # Weight by improvement magnitude (normalized)
                weights = recent_imps / (np.sum(recent_imps) + 1e-12)
                # Weighted covariance
                cov = np.zeros((d, d))
                for j in range(len(recent_dirs)):
                    w = weights[j]
                    cov += w * np.outer(recent_dirs[j], recent_dirs[j])
                # Eigendecomposition to get principal directions
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    # Sort by eigenvalue descending
                    idx = np.argsort(eigvals)[::-1]
                    # Use top 2 principal directions, complete with orthogonal vectors
                    top2 = eigvecs[:, idx[:2]]
                    # Complete the basis using QR
                    Q, R = np.linalg.qr(top2)
                    # Add orthogonal complement if needed
                    if d > 2:
                        # Generate random vectors orthogonal to Q
                        remaining = np.zeros((d, d - 2))
                        for col in range(d - 2):
                            v = rng.normal(size=d)
                            # Project out components in span of Q
                            for j in range(2):
                                v = v - np.dot(v, Q[:, j]) * Q[:, j]
                            v_norm = np.linalg.norm(v)
                            if v_norm > 1e-10:
                                remaining[:, col] = v / v_norm
                            else:
                                # Fallback to standard basis vector
                                remaining[:, col] = np.eye(d)[:, col]
                        Q_full = np.hstack([Q, remaining])
                        Q_full, _ = np.linalg.qr(Q_full)
                    else:
                        Q_full = Q
                    # Ensure right-handed system
                    if np.linalg.det(Q_full) < 0:
                        Q_full[:, 0] = -Q_full[:, 0]
                    basis = Q_full
                except np.linalg.LinAlgError:
                    pass

            # Keep history bounded
            if len(history_dirs) > 40:
                history_dirs = history_dirs[-40:]
                history_imps = history_imps[-40:]

        return x, fx
# EVOLVE-BLOCK-END
