"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start scout + L-BFGS-B polish for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input for func
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Multi-start scouting with diverse candidate points
        # For Rosenbrock, the minimum is often near (1,1,...,1) or shifted
        # Try several strategic starting points
        candidates = []
        
        # Origin
        candidates.append(np.zeros(dim))
        
        # Nominal Rosenbrock minimum (1,1,...,1)
        candidates.append(np.ones(dim))
        
        # Other strategic points
        candidates.append(np.full(dim, 0.5))
        candidates.append(np.full(dim, -1.0))
        candidates.append(np.full(dim, 2.0))
        candidates.append(np.full(dim, -2.0))
        candidates.append(np.full(dim, 3.0))
        candidates.append(np.full(dim, -3.0))
        candidates.append(np.full(dim, 4.0))
        candidates.append(np.full(dim, -4.0))
        
        # Random points within the box
        rng = np.random.default_rng(42)
        for _ in range(15):
            candidates.append(rng.uniform(-4, 4, dim))

        # Evaluate all candidates
        scored_candidates = []
        for cand in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline - 100:  # Leave time for local search
                break
            fx = safe_call(cand)
            if np.isfinite(fx):
                scored_candidates.append((fx, cand))

        # Sort by function value
        scored_candidates.sort(key=lambda x: x[0])

        # Phase 2: Local optimization from the best few starting points
        # Use L-BFGS-B which is efficient for smooth functions like Rosenbrock
        # when in the basin of attraction. Fallback to Nelder-Mead if needed.
        
        max_local_starts = min(5, len(scored_candidates))
        bounds = [(-5.0, 5.0)] * dim
        
        for i in range(max_local_starts):
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            
            fx0, x0 = scored_candidates[i]
            
            # Try L-BFGS-B first
            try:
                result = minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': max(1, budget - call_count - 10),
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                # Fallback to Nelder-Mead if L-BFGS-B fails
                try:
                    result = minimize(
                        lambda x: safe_call(x),
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': max(1, budget - call_count - 10),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final polish if budget remains
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            try:
                remaining = max(1, budget - call_count - 5)
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Return best solution found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
