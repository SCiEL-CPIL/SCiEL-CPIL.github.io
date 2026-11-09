"""Combined optimizer: k-NN surrogate with epsilon-greedy acquisition + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """k-NN surrogate optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 30s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        n = dim

        rng = np.random.default_rng(123)

        # Storage for evaluated points
        X = []  # list of points
        F = []  # list of function values
        calls_made = 0

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            X.append(x)
            F.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial quasi-random sampling for coverage
        # Use Sobol-like or Latin hypercube for initial spread
        n_initial = min(30, budget)
        # Latin hypercube sampling
        for i in range(n_initial):
            if time.time() > deadline:
                break
            # Each dimension gets a different stratified sample
            x = np.zeros(n)
            for d in range(n):
                # Stratified sampling: divide [−4, 4] into n_initial bins
                bin_idx = (i + d) % n_initial
                x[d] = -4 + 8 * (bin_idx + 0.5) / n_initial
            evaluate(x)

        # Phase 2: k-NN surrogate with epsilon-greedy acquisition
        k = 8  # number of neighbors for k-NN
        epsilon_start = 0.8
        epsilon_end = 0.1
        total_acq_steps = budget - n_initial

        for step in range(total_acq_steps):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Anneal epsilon
            progress = step / max(1, total_acq_steps)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

            # Get current data as arrays
            if len(X) < k:
                k_eff = max(2, len(X) // 2)
            else:
                k_eff = k

            X_arr = np.array(X)
            F_arr = np.array(F)

            # k-NN prediction for a candidate point
            def knn_predict(xc):
                dists = np.linalg.norm(X_arr - xc, axis=1)
                idx = np.argsort(dists)[:k_eff]
                # Inverse distance weighting
                d = dists[idx]
                d = np.maximum(d, 1e-10)
                w = 1.0 / d
                w /= np.sum(w)
                return np.dot(w, F_arr[idx])

            # Generate candidate points
            n_candidates = 50

            candidates = []

            # Exploration: maximize distance to existing points (farthest point sampling)
            if epsilon > 0:
                # Try several random points and pick the one farthest from data
                for _ in range(n_candidates // 2):
                    xc = rng.uniform(-4.5, 4.5, n)
                    dist_to_data = np.min(np.linalg.norm(X_arr - xc, axis=1))
                    candidates.append((dist_to_data, xc))

            # Exploitation: minimize k-NN predicted value
            for _ in range(n_candidates // 2):
                # Perturb existing good points
                good_idx = np.argsort(F_arr)[:max(1, len(F_arr) // 4)]
                base_idx = good_idx[rng.integers(len(good_idx))]
                xc = X_arr[base_idx] + rng.normal(0, 0.5, n)
                xc = np.clip(xc, -4.5, 4.5)
                candidates.append((0, xc))

            # Also try points near the current best
            if best_x is not None:
                for _ in range(5):
                    xc = best_x + rng.normal(0, 0.3, n)
                    xc = np.clip(xc, -4.5, 4.5)
                    candidates.append((0, xc))

            # Score candidates: mix of distance (exploration) and predicted value (exploitation)
            if len(candidates) > 0:
                # Normalize distances
                dists = np.array([c[0] for c in candidates])
                preds = np.array([knn_predict(c[1]) for c in candidates])

                # Normalize to [0,1]
                d_min, d_max = np.min(dists), np.max(dists)
                d_range = d_max - d_min + 1e-10
                d_norm = (dists - d_min) / d_range

                p_min, p_max = np.min(preds), np.max(preds)
                p_range = p_max - p_min + 1e-10
                p_norm = (preds - p_min) / p_range  # lower is better

                # Acquisition score: higher is better
                scores = epsilon * d_norm + (1 - epsilon) * (1 - p_norm)

                # Select best candidate
                best_idx = np.argmax(scores)
                xc = candidates[best_idx][1]
                evaluate(xc)
            else:
                # Fallback: random point
                xc = rng.uniform(-4, 4, n)
                evaluate(xc)

        # Phase 3: Local refinement around the best point
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Adaptive (1+1)-ES local search
            x = best_x.copy()
            sigma = 0.5
            n_local = min(50, budget - calls_made)
            for _ in range(n_local):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate a step
                z = rng.standard_normal(n)
                z /= np.linalg.norm(z) + 1e-10
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)

                f_new = evaluate(x_new)

                if f_new < best_f:
                    # Accept, keep same sigma or increase slightly
                    x = x_new.copy()
                    sigma = min(sigma * 1.2, 2.0)
                else:
                    # Reject, decrease sigma
                    sigma = max(sigma * 0.8, 0.05)

        return best_x, best_f
# EVOLVE-BLOCK-END
