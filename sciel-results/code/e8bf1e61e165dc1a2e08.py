"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 330s limit

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def scipy_wrapper(x):
            nonlocal call_count, deadline
            if call_count >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            return safe_func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Generate diverse starting points
        # For rotated Rosenbrock, the minimum is hidden but structure may persist
        # Try points near the unrotated minimum (all ones) and along the valley
        candidates = []
        
        # The unrotated Rosenbrock minimum is at (1, 1, ..., 1)
        candidates.append(np.ones(dim))
        
        # Points along the valley direction y = x^2 - 1
        for x0 in np.linspace(-3, 3, 7):
            c = np.zeros(dim)
            c[0] = x0
            for i in range(1, dim):
                c[i] = c[i-1]**2 - 1
            c = np.clip(c, -5, 5)
            candidates.append(c)
        
        # Diagonal points
        for val in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            candidates.append(np.full(dim, val))
        
        # Random points in the box
        for _ in range(20):
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        # Points near zero
        candidates.append(np.zeros(dim))
        
        # Evaluate all candidates
        candidate_results = []
        for c in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_func(np.clip(c, -5.0, 5.0))
            candidate_results.append((f, np.clip(c, -5.0, 5.0).copy()))

        if not candidate_results:
            c = np.zeros(dim)
            f = safe_func(c)
            candidate_results.append((f, c.copy()))

        # Sort by function value
        candidate_results.sort(key=lambda x: x[0])

        # Phase 2: Multi-start L-BFGS-B with gradient
        # Use finite-difference gradients for better convergence
        remaining_budget = budget - call_count
        
        # Select top candidates for optimization
        n_starts = min(len(candidate_results), 12)
        
        # Allocate budget: better starts get more
        # Use exponential weighting
        weights = [np.exp(-i * 0.3) for i in range(n_starts)]
        total_w = sum(weights)
        
        for i in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break
            
            f_init, x_init = candidate_results[i]
            
            # Allocate maxfev based on weight and remaining budget
            maxfev = max(30, int(remaining_budget * weights[i] / total_w))
            maxfev = min(maxfev, budget - call_count)
            
            if maxfev < 10:
                continue
            
            try:
                result = minimize(
                    scipy_wrapper,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': maxfev,
                        'maxfev': maxfev,
                        'gtol': 1e-10,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                
                if result.x is not None and result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
                    
            except Exception:
                pass

        # Phase 3: Final high-precision polish
        if call_count < budget - 50 and best_x is not None:
            if time.time() < deadline:
                remaining = budget - call_count
                try:
                    result = minimize(
                        scipy_wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'gtol': 1e-12,
                            'ftol': 1e-15
                        }
                    )
                    if result.x is not None and result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 4: Nelder-Mead polish if budget remains
        if call_count < budget - 30 and best_x is not None:
            if time.time() < deadline:
                remaining = budget - call_count
                try:
                    result = minimize(
                        scipy_wrapper,
                        best_x,
                        method='Nelder-Mead',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if result.x is not None and result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
