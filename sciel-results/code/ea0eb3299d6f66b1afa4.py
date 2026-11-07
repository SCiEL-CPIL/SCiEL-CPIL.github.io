# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free coordinate descent with adaptive line search that
    dynamically rotates the coordinate system using the top two principal
    directions of the recent evaluation history. This aligns search axes
    with the hidden ridge and its perpendicular normal, decoupling fast
    descent along the ridge from slow convergence across it.
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

        # Step sizes per axis (adaptive)
        steps = np.full(d, 0.5)

        # PCA update interval
        pca_interval = 15
        # Minimum number of history entries before rotating
        min_hist = 4

        # Track consecutive failures per axis for step size adaptation
        fail_counts = np.zeros(d, dtype=int)
        success_counts = np.zeros(d, dtype=int)

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
                found_improvement = False

                # Try positive and negative directions with multiple step sizes
                for sign in [1, -1]:
                    for mult in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                        if calls >= B:
                            break
                        trial = x + sign * steps[k] * mult * direction
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls += 1
                        if ft < best_f:
                            best_f = ft
                            best_x = trial
                            found_improvement = True

                if found_improvement:
                    # Record the move
                    move = best_x - x
                    norm = np.linalg.norm(move)
                    if norm > 1e-12:
                        history_dirs.append(move / norm)
                        history_imps.append(fx - best_f)
                    x = best_x
                    fx = best_f
                    # Increase step size on success
                    steps[k] = min(steps[k] * 1.25, 5.0)
                    success_counts[k] += 1
                    fail_counts[k] = 0
                else:
                    # No improvement; shrink step
                    steps[k] = max(steps[k] * 0.7, 0.01)
                    fail_counts[k] += 1

            # Periodically update the basis using PCA of recent history
            if len(history_dirs) >= min_hist and (i + 1) % pca_interval == 0:
                # Take the most recent history entries
                n_recent = min(len(history_dirs), 10)
                recent_dirs = np.array(history_dirs[-n_recent:])
                recent_imps = np.array(history_imps[-n_recent:])
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
                    new_basis = eigvecs[:, idx]
                    # Ensure the basis is orthogonal and has correct orientation
                    # Use QR decomposition to orthonormalize
                    Q, R = np.linalg.qr(new_basis)
                    # Ensure right-handed system
                    if np.linalg.det(Q) < 0:
                        Q[:, 0] = -Q[:, 0]
                    basis = Q
                    # Reset step sizes when basis changes significantly
                    steps = np.full(d, 0.5)
                    fail_counts = np.zeros(d, dtype=int)
                    success_counts = np.zeros(d, dtype=int)
                except np.linalg.LinAlgError:
                    pass

            # Keep history bounded
            if len(history_dirs) > 40:
                history_dirs = history_dirs[-40:]
                history_imps = history_imps[-40:]

        return x, fx
# EVOLVE-BLOCK-END
