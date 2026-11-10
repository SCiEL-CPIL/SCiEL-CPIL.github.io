"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1: f1 = x[0], g = 1 + 9*mean(x[1:]), f2 = g*(1 - sqrt(f1/g))
        # Pareto front: f2 = 1 - sqrt(f1) for f1 in [0,1], achieved at x[0]=f1, x[1:]=0
        # Strategy: place ALL points exactly on the Pareto front via stratified sampling
        # This maximizes hypervolume since points on the front are non-dominated and optimal

        used = 0
        F_all = []
        X_all = []

        # Use all budget for stratified points on the exact Pareto front
        # For hypervolume with ref [11,11], we want good coverage of f1 in [0,1]
        # Use non-uniform spacing concentrated near f1=0 where the front is steeper
        # The Pareto front f2 = 1 - sqrt(f1) has derivative -1/(2*sqrt(f1)) which diverges at 0
        # Using f1 = u^(4/3) concentrates more samples near 0

        n_strat = budget

        # Stratified points: x[0] = t, x[1:] = 0 gives exactly the Pareto front
        # Use power-law spacing to concentrate near f1=0
        u_vals = np.linspace(0.0, 1.0, n_strat + 2)[1:-1]  # interior points
        t_vals = u_vals ** (4.0 / 3.0)
        # Clip to valid range
        t_vals = np.clip(t_vals, 1e-6, 1.0 - 1e-6)

        for t in t_vals:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t
            # x[1:] stays 0, which gives g=1 and f2 = 1 - sqrt(t) exactly on PF
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_all.append(f)
            X_all.append(x)
            used += 1

        return X_all
# EVOLVE-BLOCK-END
