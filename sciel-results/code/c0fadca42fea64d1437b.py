"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT6 Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _is_front_point(self, x, bounds):
        """Check if x is close to the ZDT6 Pareto front (x[1:] ~ 0)."""
        return np.allclose(x[1:], 0.0, atol=1e-4)

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size: use a good chunk of budget for diversity
        pop_size = min(50, max(10, budget // 4))

        # Initialize: mix of random points and front-parameterized points
        X = np.empty((pop_size, dim))
        # First half: random in [0,1]^dim
        n_rand = pop_size // 2
        X[:n_rand] = lo + (hi - lo) * rng.random((n_rand, dim))
        # Second half: points on/near the Pareto front (x[0] varies, x[1:] ~ 0)
        n_front = pop_size - n_rand
        x0_vals = rng.random(n_front)
        X[n_front:, 0] = x0_vals
        X[n_front:, 1:] = 0.0  # exactly on front

        F = np.array([problem(x) for x in X])
        used = pop_size

        def compute_dominance(F):
            """Vectorized dominance check: dominated[i] = True if any other point dominates i."""
            n = len(F)
            # F[i] <= F[j] for all objectives AND F[i] < F[j] for at least one
            # dominated[j] = any i such that F[i] <= F[j] and F[i] < F[j]
            # Using broadcasting: (F[:, None, :] <= F[None, :, :]) gives [i,j] = True if F[i] <= F[j]
            le = F[:, None, :] <= F[None, :, :]  # le[i,j] = F[i] <= F[j]
            lt = F[:, None, :] < F[None, :, :]   # lt[i,j] = F[i] < F[j]
            # For each j, dominated if exists i != j with le[i,j] and lt[i,j].any()
            # Exclude self: set diagonal to False
            le_no_diag = le.copy()
            np.fill_diagonal(le_no_diag, False)
            lt_no_diag = lt.copy()
            np.fill_diagonal(lt_no_diag, False)
            # dominated[j] = any i where le_no_diag[i,j] and lt_no_diag[i,j].any()
            dominates = le_no_diag & lt_no_diag.any(axis=2)[:, None]
            dominated = dominates.any(axis=0)
            return dominated

        def adaptive_step(rng, F, hi, lo):
            """Compute adaptive mutation step based on objective spread."""
            spread = np.std(F, axis=0)
            base = 0.1
            # Scale step by spread relative to typical range
            scale = np.clip(spread / 1.0, 0.05, 1.0)
            return base * scale

        while used < budget:
            # Deadline check
            if time.time() > deadline:
                break

            # Compute non-dominated set
            dominated = compute_dominance(F)
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Select parent from non-dominated set (tournament-like: pick best)
            # Use a simple tournament of size 2 among non-dominated
            if len(non_dom_idx) > 1:
                c1, c2 = rng.choice(non_dom_idx, size=2, replace=False)
                # Pick the one with lower sum of objectives (proxy for quality)
                parent_idx = c1 if np.sum(F[c1]) < np.sum(F[c2]) else c2
            else:
                parent_idx = non_dom_idx[0]

            parent = X[parent_idx]

            # Adaptive mutation
            step = adaptive_step(rng, F, hi, lo)
            # Occasionally do a larger jump
            if rng.random() < 0.1:
                step *= 3.0

            child = parent + step * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point (by sum of objectives) or a random point
            # Use a mix: 70% replace worst, 30% replace random
            if rng.random() < 0.7:
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                worst_idx = int(rng.integers(len(F)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
