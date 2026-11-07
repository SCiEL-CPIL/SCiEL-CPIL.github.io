"""New optimizer using Multi-Start Nelder-Mead Simplex Method."""
import numpy as np


class Optimizer:
    """Nelder-Mead style continuous optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x_global = np.zeros(dim)
        best_fx_global = np.inf

        # Number of restarts
        n_restarts = 3
        budget_per_restart = budget // n_restarts
        if budget_per_restart < 20:
            n_restarts = 1
            budget_per_restart = budget

        for r in range(n_restarts):
            # Initialize center
            if r == 0:
                center = np.zeros(dim)
            else:
                center = rng.uniform(-4.0, 4.0, dim)
            
            # Initialize simplex
            delta = 2.0
            simplex = [center.copy()]
            fvals = [func(center)]
            
            for i in range(dim):
                p = center.copy()
                p[i] += delta
                p = np.clip(p, -5, 5)
                simplex.append(p)
                fvals.append(func(p))
            
            calls_made = dim + 1
            remaining = budget_per_restart - calls_made
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while remaining > 0:
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Convergence check
                if max(fvals) - min(fvals) < 1e-6:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                remaining -= 1
                calls_made += 1
                if remaining < 0: break
                
                if fr < fvals[0]:
                    # Expansion
                    if remaining <= 0: break
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    remaining -= 1
                    calls_made += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if remaining <= 0: break
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    remaining -= 1
                    calls_made += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if remaining <= 0: break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            remaining -= 1
                            calls_made += 1

                # Update global best
                if fvals[0] < best_fx_global:
                    best_fx_global = fvals[0]
                    best_x_global = simplex[0].copy()

        return best_x_global, best_fx_global
