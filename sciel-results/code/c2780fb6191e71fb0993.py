# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Hybrid L-BFGS-B + Nelder-Mead for Bent Cigar's anisotropic valley."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0

        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: L-BFGS-B multi-restart (coarse search)
        # Use fewer, more targeted restarts
        n_lbgfs_restarts = 6

        for i in range(n_lbgfs_restarts):
            if calls >= budget - 30:
                break
            if time.time() > deadline:
                break

            if i == 0:
                x_start = np.zeros(dim)
            elif i == 1:
                x_start = np.array([4.0] * dim)
            elif i == 2:
                x_start = np.array([-4.0] * dim)
            else:
                x_start = rng.uniform(-5, 5, dim)

            try:
                maxfev = min(80, max(20, budget - calls - 50))
                result = scipy.optimize.minimize(
                    wrapper,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 60, 'maxfev': maxfev,
                             'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish on the best point found
        if calls < budget - 10 and time.time() < deadline:
            try:
                # Initialize NM simplex around best_x
                x0 = best_x.copy()
                # Small initial step size for refinement
                init_step = 0.5

                def nm_wrapper(x):
                    nonlocal calls
                    if calls >= budget:
                        return float('inf')
                    if time.time() > deadline:
                        return float('inf')
                    x_clipped = np.clip(x, -5, 5)
                    calls += 1
                    try:
                        return float(func(x_clipped))
                    except Exception:
                        return float('inf')

                nm_result = scipy.optimize.minimize(
                    nm_wrapper,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(100, budget - calls - 5),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'initial_simplex': None,
                        'adaptive': True
                    }
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = np.clip(nm_result.x, -5, 5)
            except Exception:
                pass

        # Phase 3: If still have budget, try a few more NM polishes from
        # slightly perturbed best_x
        if calls < budget - 5 and time.time() < deadline:
            for perturb in [0.2, -0.2, 0.1]:
                if calls >= budget - 5 or time.time() > deadline:
                    break
                try:
                    x0_pert = best_x + perturb * rng.standard_normal(dim) * 0.3
                    x0_pert = np.clip(x0_pert, -5, 5)

                    def nm_wrapper2(x):
                        nonlocal calls
                        if calls >= budget:
                            return float('inf')
                        if time.time() > deadline:
                            return float('inf')
                        x_clipped = np.clip(x, -5, 5)
                        calls += 1
                        try:
                            return float(func(x_clipped))
                        except Exception:
                            return float('inf')

                    nm_result2 = scipy.optimize.minimize(
                        nm_wrapper2,
                        x0_pert,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': min(50, budget - calls - 5),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if nm_result2.fun < best_f:
                        best_f = nm_result2.fun
                        best_x = np.clip(nm_result2.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
