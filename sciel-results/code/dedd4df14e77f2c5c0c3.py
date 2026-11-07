"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a combination of space-filling design and
    greedy distance-based selection in objective space, with a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim):
        """Generate n points in the unit hypercube using the Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # primes: 2, 3, 5, 7, ...
            inv_base = 1.0 / base
            f = inv_base
            for i in range(n):
                points[i, d] = f
                f = inv_base * f
                while f >= 1.0:
                    f = f / base - np.floor(f / base)
                    f = inv_base * f
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Space-filling initial design using Halton sequence
        n_init = min(50, self.budget // 2)
        halton = self._halton_sequence(n_init, self.dim)
        X_init = lo + span * halton
        F = np.array([problem(x) for x in X_init])
        used = n_init

        # Store all evaluated points
        all_X = list(X_init)
        all_F = list(F)

        # Phase 2: Greedy distance-based selection in objective space
        # For each candidate, compute minimum distance to existing points in objective space
        # Select the candidate that maximizes this minimum distance (farthest-point sampling)

        # Generate a pool of candidate points for selection
        # Use random candidates plus some structured ones
        n_candidates_pool = 50
        candidates = []

        # Add some random candidates
        rng = np.random.default_rng(42)
        for _ in range(n_candidates_pool):
            c = lo + span * rng.random(self.dim)
            candidates.append(c)

        # Add some points near the "corners" of the objective space
        for _ in range(n_candidates_pool // 2):
            c = lo + span * rng.random(self.dim)
            # Perturb towards a random vertex
            vertex = np.random.choice([lo, hi], self.dim, replace=True)
            c = 0.5 * c + 0.5 * vertex
            c = np.clip(c, lo, hi)
            candidates.append(c)

        # For each remaining budget evaluation, select the best candidate
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute minimum distance from each candidate to all existing points in objective space
            # We need to evaluate candidates to get their objective values, but that costs budget.
            # Instead, use a surrogate: project candidates to objective space using a simple model,
            # or just pick the candidate that is farthest in decision space from existing points.
            
            # Actually, we need to evaluate to know the objective values. Let's use a different strategy:
            # Pick the candidate that is farthest in decision space from the existing points.
            
            # Compute distances from each candidate to all existing points in decision space
            if not all_X:
                idx = 0
            else:
                X_arr = np.array(all_X)
                dists = np.array([np.min(np.linalg.norm(X_arr - c, axis=1)) for c in candidates])
                idx = np.argmax(dists)

            x_new = candidates[idx]
            f_new = problem(x_new)
            used += 1
            all_X.append(x_new)
            all_F.append(f_new)

            # Update candidate pool: remove the selected candidate and add new random ones
            candidates.pop(idx)
            for _ in range(2):
                c = lo + span * rng.random(self.dim)
                candidates.append(c)

        return all_X
# EVOLVE-BLOCK-END
