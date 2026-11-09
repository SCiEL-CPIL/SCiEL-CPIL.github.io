"""Rewritten optimizer using deterministic Sobol sampling + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import time


class Optimizer:
    """Model-based optimizer: Sobol global sampling then Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # Call counter to track budget
        call_count = [0]

        def safe_func(x):
            if call_count[0] >= budget:
                return float('inf')
            call_count[0] += 1
            return func(x)

        # Phase 1: Sobol sequence sampling (deterministic, systematic coverage)
        # Use 200 points for initial exploration
        n_sobol = min(200, budget - 100)  # leave room for local refinement
        sampler = qmc.Sobol(d=dim, scramble=False, seed=42)
        sobol_points_unit = sampler.random(n_sobol)
        # Map from [0,1]^dim to [-5,5]^dim
        sobol_points = sobol_points_unit * 10.0 - 5.0

        best_x = None
        best_f = float('inf')

        for pt in sobol_points:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            f_val = safe_func(pt)
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Phase 2: Nelder-Mead refinement from best point
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget - call_count[0],
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains and time allows, try a few more local refinements
        # from other good Sobol points (top 5)
        if call_count[0] < budget and time.time() < deadline:
            # Re-evaluate: we need to track which points were good
            # Since we only kept best, let's do a second pass with more points
            # Actually, let's just do a few more Nelder-Mead from perturbed best
            for trial in range(3):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                # Perturb the best point slightly
                perturbation = np.random.default_rng(42 + trial).uniform(-0.5, 0.5, dim)
                start_point = np.clip(best_x + perturbation, -5, 5)
                try:
                    result = minimize(
                        safe_func,
                        start_point,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(1, budget - call_count[0]),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
