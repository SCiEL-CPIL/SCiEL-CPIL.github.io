"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Global Optimization using Sobol sequence with adaptive local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Sobol global exploration
        # Use Sobol sequence for low-discrepancy sampling
        # Reserve some budget for local refinement
        n_sobol = int(budget * 0.8)  # 80% for global
        n_local = budget - n_sobol   # 20% for local refinement

        # Generate Sobol samples
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_sobol)

        # Map to [-5, 5]^dim
        sobol_mapped = sobol_samples * 10.0 - 5.0

        # Evaluate all Sobol points
        best_x = None
        best_f = np.inf
        all_points = []
        all_values = []

        for i in range(n_sobol):
            x = sobol_mapped[i]
            fx = func(x)
            all_points.append(x)
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement around the best point found
        # Use a small population-based local search (CMA-like or simple ES)
        # Start with a small neighborhood around best_x

        if n_local > 0:
            # Deterministic local refinement using Coordinate Descent
            x_best = best_x.copy()
            f_best = best_f
            step = 1.0
            remaining = n_local

            while remaining >= 2 * dim:
                improved = False
                for d in range(dim):
                    if remaining < 2:
                        break
                    
                    # Try positive step
                    x_pos = x_best.copy()
                    x_pos[d] += step
                    if x_pos[d] > 5.0: x_pos[d] = 5.0
                    f_pos = func(x_pos)
                    remaining -= 1
                    
                    # Try negative step
                    x_neg = x_best.copy()
                    x_neg[d] -= step
                    if x_neg[d] < -5.0: x_neg[d] = -5.0
                    f_neg = func(x_neg)
                    remaining -= 1
                    
                    # Choose best move
                    if f_pos < f_best:
                        x_best = x_pos
                        f_best = f_pos
                        improved = True
                    elif f_neg < f_best:
                        x_best = x_neg
                        f_best = f_neg
                        improved = True
                    else:
                        # No improvement in this dimension, shrink step
                        step *= 0.5
                        if step < 1e-4:
                            break

                if step < 1e-4:
                    break

            best_x = x_best
            best_f = f_best

        return best_x, best_f
# EVOLVE-BLOCK-END
