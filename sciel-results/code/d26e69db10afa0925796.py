"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


class Optimizer:
    """Surrogate-assisted optimizer with GP-like model and EI acquisition for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Broad random search to get a decent starting point
        n_init = min(30, budget // 10)
        points = []
        values = []
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: Surrogate-assisted optimization with EI acquisition
        # Use a simple distance-based surrogate (like a weighted nearest neighbor)
        # with expected improvement acquisition

        calls_used = n_init
        stuck_counter = 0

        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            points_arr = np.array(points)
            values_arr = np.array(values)

            # Build simple surrogate: inverse distance weighted prediction
            # For each candidate, predict value using nearby points
            n_candidates = 20
            candidates = []

            # Mix of exploration and exploitation candidates
            # Exploitation: near best point
            for i in range(n_candidates // 2):
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                candidates.append(x)

            # Exploration: random points in unexplored regions
            for i in range(n_candidates // 2):
                x = rng.uniform(-5, 5, dim)
                candidates.append(x)

            # Evaluate candidates using surrogate to pick best EI
            # Simple EI: for each candidate, estimate improvement over best_f
            # using weighted average of nearby points
            best_ei = -np.inf
            best_candidate = None

            for cand in candidates:
                # Compute distances to all evaluated points
                dists = np.linalg.norm(points_arr - cand, axis=1)
                # Inverse distance weights (with small epsilon to avoid division by zero)
                weights = 1.0 / (dists + 0.1)
                weights /= weights.sum()

                # Predicted value
                pred_val = np.dot(weights, values_arr)

                # Uncertainty: spread of nearby values
                uncertainty = np.sqrt(np.dot(weights, (values_arr - pred_val)**2)) + 0.1

                # Expected Improvement
                improvement = best_f - pred_val
                if improvement > 0:
                    ei = improvement + 0.1 * uncertainty
                else:
                    ei = -0.1 * uncertainty

                if ei > best_ei:
                    best_ei = ei
                    best_candidate = cand

            if best_candidate is None:
                best_candidate = rng.uniform(-5, 5, dim)

            # Evaluate the best candidate
            x_new = best_candidate
            fx_new = func(x_new)
            calls_used += 1

            points.append(x_new)
            values.append(fx_new)

            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1

            # If stuck for a while, do a random restart
            if stuck_counter > 50 and calls_used < budget * 0.9:
                stuck_counter = 0
                for _ in range(5):
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_used += 1
                    points.append(x)
                    values.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return (best_x, best_f)
# EVOLVE-BLOCK-END
