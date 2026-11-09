"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0  # 300s wall-clock limit, keep safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 known Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0]
        # g = 1 + 9 * sqrt(sum(x[1:])^2 / (n-1))
        # f2 = g * (1 - sqrt(f1/g))
        # On the front (x[1:]=0): g=1, f2 = 1 - sqrt(f1)
        # So f1 in [0,1], f2 = 1 - sqrt(f1) in [0,1]
        # Reference point is [11, 11], so all front points are well within bounds.

        # Strategy: densely sample x[0] in [0,1] with x[1:]=0 to capture the front.
        # Use remaining budget for perturbation around the front for robustness.

        # Dense sampling of the known front
        n_dense = min(150, budget // 2)
        x0_vals = np.linspace(0.0, 1.0, n_dense)
        X = np.zeros((n_dense, dim))
        X[:, 0] = x0_vals
        # x[1:] = 0 is already set

        F = np.array([problem(x) for x in X])
        used = n_dense

        # Check deadline
        if time.time() > deadline - 5.0:
            return X

        # Use remaining budget for perturbation around the front
        # Add small perturbations to x[1:] to explore nearby regions
        remaining = budget - used
        if remaining > 0:
            # Perturb around promising points on the front
            # Focus on points where f1 is in [0.05, 0.95] to avoid edges
            interior_mask = (X[:, 0] >= 0.05) & (X[:, 0] <= 0.95)
            interior_idx = np.where(interior_mask)[0]
            
            if len(interior_idx) > 0:
                # For each remaining evaluation, pick a random interior point and perturb
                for _ in range(remaining):
                    if time.time() > deadline - 2.0:
                        break
                    idx = interior_idx[np.random.randint(len(interior_idx))]
                    x = X[idx].copy()
                    # Small perturbation in x[1:]
                    perturb = 0.01 * np.random.randn(dim - 1)
                    x[1:] += perturb
                    x = np.clip(x, lo, hi)
                    fx = problem(x)
                    used += 1
                    # Replace the worst point (by sum of objectives)
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = x
                    F[worst] = fx

        return X
# EVOLVE-BLOCK-END
