"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using structured initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 structure: x[0] controls front position, x[1:] should be near 0
        # Pareto front: f1 = x[0], f2 = (1-x[0])*cos(x[0]*pi/2), f3 = (1-x[0])*sin(x[0]*pi/2)
        # when g=1 (i.e., x[1:]=0)

        # Phase 1: Structured initialization along the expected front
        n_init = min(50, budget // 4)
        X_list = []
        F_list = []

        for i in range(n_init):
            x = np.zeros(dim)
            x[0] = i / max(n_init - 1, 1)  # spread along first variable
            # Add small noise to x[1:] to ensure g is close to 1 but not exactly
            x[1:] = rng.uniform(0, 0.01, dim - 1)
            x = np.clip(x, lo, hi)
            X_list.append(x)
            F_list.append(problem(x))

        used = n_init

        # Phase 2: Local refinement - perturb points near the front
        # Focus on reducing x[1:] to get g closer to 1
        while used < budget:
            if time.time() > deadline:
                break

            # Pick a random point from current set
            idx = rng.integers(len(X_list))
            parent = X_list[idx].copy()

            # Strategy: reduce x[1:] components (push toward 0)
            # and adjust x[0] slightly
            child = parent.copy()

            # Reduce x[1:] toward 0 with some randomness
            for j in range(1, dim):
                if child[j] > 0:
                    # Move toward 0 with some noise
                    reduction = rng.uniform(0.5, 1.0)
                    child[j] = child[j] * (1 - reduction) + rng.uniform(-0.005, 0.005)
                    child[j] = max(0.0, min(1.0, child[j]))

            # Slightly perturb x[0] to explore nearby front positions
            child[0] = parent[0] + rng.uniform(-0.05, 0.05)
            child[0] = max(0.0, min(1.0, child[0]))

            fc = problem(child)
            used += 1

            # Replace worst point or add if beneficial
            # Use a simple criterion: replace point with highest sum of objectives
            if len(F_list) >= n_init:
                sums = np.sum(np.array(F_list), axis=1)
                worst_idx = int(np.argmax(sums))
                # Only replace if child is better
                if np.sum(fc) < sums[worst_idx]:
                    X_list[worst_idx] = child
                    F_list[worst_idx] = fc
            else:
                X_list.append(child)
                F_list.append(fc)

        return X_list
# EVOLVE-BLOCK-END
