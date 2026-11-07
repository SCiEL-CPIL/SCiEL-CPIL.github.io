"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead ensemble with Latin Hypercube initial points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Generate Latin Hypercube sample points for 5 diverse starts
        n_starts = 5
        lhs_points = np.zeros((n_starts, dim))
        for d in range(dim):
            perm = rng.permutation(n_starts)
            for i in range(n_starts):
                idx = perm[i]
                low = -5.0 + (10.0 / n_starts) * idx
                high = -5.0 + (10.0 / n_starts) * (idx + 1)
                lhs_points[i, d] = rng.uniform(low, high)

        # Track best solution and call count
        state = {
            'calls': 0,
            'best_x': None,
            'best_f': np.inf
        }

        def counting_func(x):
            state['calls'] += 1
            f = func(x)
            if f < state['best_f']:
                state['best_f'] = f
                state['best_x'] = np.array(x)
            return f

        # Evaluate LHS points
        for i in range(n_starts):
            x0 = lhs_points[i]
            counting_func(x0)

        # Run Nelder-Mead from each start
        for i in range(n_starts):
            if state['calls'] >= budget:
                break
            
            x0 = lhs_points[i]
            
            try:
                minimize(
                    counting_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Use remaining budget for final refinement of best point if found
        if state['best_x'] is not None and state['calls'] < budget:
            remaining = budget - state['calls']
            if remaining > 10:
                try:
                    minimize(
                        counting_func,
                        state['best_x'],
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(1, remaining // 6),
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    pass

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
