"""Multi-start L-BFGS-B with final high-precision polish for BBOB f12 (bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start L-BFGS-B with final polish for BBOB f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def check_budget():
            nonlocal total_calls
            if total_calls >= budget or time.time() > deadline:
                return False
            return True

        from scipy.optimize import minimize

        # First, evaluate center point
        center = np.zeros(dim)
        f_center = func(center)
        total_calls += 1
        best_f = f_center
        best_x = center.copy()

        # Phase 1: Multi-start L-BFGS-B exploration
        # Use more restarts with fewer evals each for better coverage
        # Bent cigar is highly anisotropic; more diverse starts help find the valley
        n_restarts = 15
        remaining = budget - 1
        evals_per_restart = max(20, remaining // n_restarts)
        # Reserve some budget for final polish
        polish_budget = min(100, remaining // 5)
        explore_budget = remaining - polish_budget
        evals_per_restart = max(15, explore_budget // n_restarts)

        for r in range(n_restarts):
            if not check_budget():
                break
            
            # Diverse starting points: mix of uniform, scaled, and axis-aligned
            if r == 0:
                x0 = rng.uniform(-3.0, 3.0, dim)
            elif r == 1:
                x0 = rng.uniform(-4.5, 4.5, dim)
            elif r == 2:
                # Try points along axes
                axis = rng.integers(0, dim)
                x0 = np.zeros(dim)
                x0[axis] = rng.uniform(-4.5, 4.5)
            else:
                x0 = rng.uniform(-4.5, 4.5, dim)
            
            # Count function evaluations for this run
            def objective(x):
                nonlocal total_calls, best_x, best_f
                if total_calls >= budget or time.time() > deadline:
                    return best_f  # Return current best to stop optimization
                total_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            # Run L-BFGS-B with tighter tolerances
            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 60, 'maxfun': evals_per_restart, 'gtol': 1e-8}
                )
                # Update best from result
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Final high-precision polish on the best solution
        # This uses a tight tolerance and more iterations for a precise local optimum
        if check_budget():
            def polish_objective(x):
                nonlocal total_calls, best_x, best_f
                if total_calls >= budget or time.time() > deadline:
                    return best_f
                total_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(
                    polish_objective,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': polish_budget, 'gtol': 1e-12, 'ftol': 1e-15}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a Nelder-Mead polish for final precision
        remaining_calls = budget - total_calls
        if remaining_calls > 20 and time.time() < deadline - 5.0:
            def nm_objective(x):
                nonlocal total_calls, best_x, best_f
                if total_calls >= budget or time.time() > deadline:
                    return best_f
                total_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                # Small initial simplex for NM polish
                x_best = best_x.copy()
                simplex = [x_best]
                for i in range(dim):
                    pert = x_best.copy()
                    step = 0.01 if abs(x_best[i]) < 1.0 else 0.1
                    pert[i] += step
                    simplex.append(pert)
                
                res = minimize(
                    nm_objective,
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': min(200, remaining_calls), 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
