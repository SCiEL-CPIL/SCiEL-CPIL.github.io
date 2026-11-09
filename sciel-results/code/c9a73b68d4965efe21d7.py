"""Black-box optimizer for BBOB f14 (Different powers): Sobol + Nelder-Mead + BFGS."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Sobol global search followed by Nelder-Mead and BFGS local refinement."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 260.0  # safety margin under 300 s limit

        # --- Phase 1: Sobol global exploration ---
        # Use a Sobol sequence for structured coverage of [-5, 5]^dim
        n_sobol = min(32, budget // 3)  # leave budget for local refinement
        if n_sobol < 8:
            n_sobol = 8

        # Generate Sobol points
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sobol.random(n=n_sobol)  # in [0, 1]^dim
        sobol_points = sobol_points * 10.0 - 5.0  # map to [-5, 5]^dim

        best_f = np.inf
        best_x = None
        used = 0

        for pt in sobol_points:
            if used >= budget or time.time() > deadline:
                break
            f_val = func(pt)
            used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # --- Phase 2: Nelder-Mead local refinement from best Sobol point ---
        if used < budget and time.time() < deadline:
            nm_maxiter = min(200, (budget - used) // 2)
            nm_result = minimize(
                func, best_x, method='Nelder-Mead',
                options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True, 'initial_simplex': None}
            )
            # Count approximate function calls for Nelder-Mead
            # Nelder-Mead doesn't easily report call count, so we estimate
            # For dim=5, each iteration uses ~dim+1 calls, so ~6*maxiter calls
            nm_calls = min(int(6 * nm_result.nit), budget - used)
            used += nm_calls

            if nm_result.fun < best_f:
                best_f = nm_result.fun
                best_x = nm_result.x.copy()

        # --- Phase 3: BFGS local refinement (quasi-Newton for smooth convergence) ---
        if used < budget and time.time() < deadline:
            bfgs_maxiter = min(100, (budget - used) // 2)
            bfgs_result = minimize(
                func, best_x, method='BFGS',
                options={'maxiter': bfgs_maxiter, 'gtol': 1e-8}
            )
            # BFGS uses ~dim calls per iteration for gradient approximation
            bfgs_calls = min(int(dim * bfgs_result.nit + 1), budget - used)
            used += bfgs_calls

            if bfgs_result.fun < best_f:
                best_f = bfgs_result.fun
                best_x = bfgs_result.x.copy()

        # --- Phase 4: If budget remains, try multi-start Nelder-Mead from other Sobol points ---
        if used < budget and time.time() < deadline:
            remaining = budget - used
            # Try up to 3 additional Nelder-Mead starts from next-best Sobol points
            sorted_sobol = sorted(
                [(func(pt), pt) for pt in sobol_points if not np.array_equal(pt, best_x)],
                key=lambda x: x[0]
            )[:3]
            for f_init, x_init in sorted_sobol:
                if used >= budget or time.time() > deadline:
                    break
                nm_maxiter = min(100, (budget - used) // 3)
                nm_result = minimize(
                    func, x_init, method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-10,
                             'adaptive': True}
                )
                nm_calls = min(int(6 * nm_result.nit), budget - used)
                used += nm_calls

                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = nm_result.x.copy()

        # Clip final result to box
        if best_x is not None:
            best_x = np.clip(best_x, -5.0, 5.0)
            # Final evaluation to ensure validity
            final_f = func(best_x)
            if final_f < best_f:
                best_f = final_f

        return best_x, best_f
# EVOLVE-BLOCK-END
