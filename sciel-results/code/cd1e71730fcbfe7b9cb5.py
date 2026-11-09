"""Katsuura (f23) minimizer using Multi-Start CMA-ES with adaptive restarts.

Strategy:
  1. Initial coarse sampling to identify promising regions.
  2. Multi-start CMA-ES: run several independent CMA-ES instances from different
     starting points. CMA-ES is robust to multimodality and makes efficient use
     of the call budget.
  3. Adaptive restarts: if a CMA-ES instance stagnates, restart from a new point.
  4. Final polish with tight CMA-ES on the best point found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Katsuura (f23) in 5 dimensions using Multi-Start CMA-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        state = {'calls': 0, 'best_x': None, 'best_f': float('inf')}
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit

        def call(x):
            """Evaluate one point, update best, respect budget and time."""
            if state['calls'] >= budget or not time_ok():
                return state['best_f']
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return state['best_f']
            if not np.isfinite(v):
                v = state['best_f']
                return v
            state['calls'] += 1
            if v < state['best_f']:
                state['best_f'] = v
                state['best_x'] = x.copy()
            return v

        # ---------- 1. Initial coarse sampling ----------
        probes = [np.zeros(dim)]
        # Uniform random points
        for _ in range(30):
            probes.append(rng.uniform(-5, 5, dim))
        # Gaussian around origin
        for _ in range(15):
            probes.append(np.clip(rng.normal(0, 1.5, dim), -5, 5))

        for p in probes:
            if state['calls'] >= budget - 100 or not time_ok():
                break
            call(p)

        best = state['best_x'].copy() if state['best_x'] is not None else np.zeros(dim)

        # ---------- 2. Multi-Start CMA-ES ----------
        try:
            from cma import CMA
        except ImportError:
            CMA = None

        if CMA is not None:
            # Run multiple CMA-ES instances
            n_starts = 6
            # Allocate calls more aggressively to early starts
            remaining_budget = budget - state['calls'] - 30
            # Give first start more budget, subsequent starts less
            start_budgets = []
            if remaining_budget > 0:
                # First start gets 40%, rest split among remaining
                first_budget = int(remaining_budget * 0.4)
                rest_budget = remaining_budget - first_budget
                per_start = max(20, rest_budget // (n_starts - 1))
                start_budgets.append(first_budget)
                for _ in range(n_starts - 1):
                    start_budgets.append(per_start)
            else:
                start_budgets = [0] * n_starts

            for start_idx in range(n_starts):
                if not time_ok() or state['calls'] >= budget - 10:
                    break
                
                # Starting point: best found for first, then random
                if start_idx == 0:
                    x0 = best.copy()
                    sigma0 = 0.5
                else:
                    x0 = rng.uniform(-5, 5, dim)
                    sigma0 = 2.0
                
                # CMA-ES parameters
                bounds = [(-5.0, 5.0)] * dim
                max_calls_this_start = start_budgets[start_idx] if start_idx < len(start_budgets) else 50
                
                try:
                    es = CMA(x0, sigma0, {
                        'bounds': bounds,
                        'maxiter': 2000,
                        'seed': start_idx + 100,
                        'verbose': -9,
                        'tolopt': 1e-13,
                        'tolfun': 1e-13,
                        'tolfunint': 1e-13,
                        'popsize': 8,  # smaller population for efficiency
                    })
                    
                    calls_this_start = 0
                    while not es.stop() and state['calls'] < budget - 10 and time_ok() and calls_this_start < max_calls_this_start:
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 10 or not time_ok() or calls_this_start >= max_calls_this_start:
                                break
                            v = call(sol)
                            fitnesses.append(v)
                            calls_this_start += 1
                        
                        if len(fitnesses) < len(solutions):
                            # Pad with worst fitness if we ran out of budget
                            worst = max(fitnesses) if fitnesses else state['best_f']
                            fitnesses.extend([worst] * (len(solutions) - len(fitnesses)))
                        
                        es.tell(solutions, fitnesses)
                    
                    # Update best from CMA-ES result
                    if es.result.xbest is not None:
                        xb = np.clip(np.asarray(es.result.xbest, dtype=float), -5, 5)
                        fb = call(xb)
                        if fb < state['best_f']:
                            state['best_f'] = fb
                            state['best_x'] = xb.copy()
                            best = xb.copy()
                            
                except Exception:
                    pass

        else:
            # Fallback: use scipy Nelder-Mead if CMA-ES not available
            from scipy.optimize import minimize
            n_starts = 4
            for start_idx in range(n_starts):
                if not time_ok() or state['calls'] >= budget - 30:
                    break
                x0 = best.copy() if start_idx == 0 else rng.uniform(-5, 5, dim)
                try:
                    res = minimize(
                        lambda x: call(x),
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                 'adaptive': True},
                    )
                    if np.isfinite(res.fun) and res.fun < state['best_f']:
                        state['best_f'] = float(res.fun)
                        state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
                        best = state['best_x'].copy()
                except Exception:
                    pass

        # ---------- 3. Final polish with tight CMA-ES ----------
        if time_ok() and state['calls'] < budget - 10:
            if CMA is not None:
                try:
                    sigma0 = 0.05
                    bounds = [(-5.0, 5.0)] * dim
                    es = CMA(best.copy(), sigma0, {
                        'bounds': bounds,
                        'maxiter': 1000,
                        'seed': 999,
                        'verbose': -9,
                        'tolopt': 1e-15,
                        'tolfun': 1e-15,
                        'tolfunint': 1e-15,
                        'popsize': 6,
                    })
                    
                    while not es.stop() and state['calls'] < budget - 5 and time_ok():
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 5 or not time_ok():
                                break
                            v = call(sol)
                            fitnesses.append(v)
                        
                        if len(fitnesses) < len(solutions):
                            worst = max(fitnesses) if fitnesses else state['best_f']
                            fitnesses.extend([worst] * (len(solutions) - len(fitnesses)))
                        
                        es.tell(solutions, fitnesses)
                    
                    if es.result.xbest is not None:
                        xb = np.clip(np.asarray(es.result.xbest, dtype=float), -5, 5)
                        fb = call(xb)
                        if fb < state['best_f']:
                            state['best_f'] = fb
                            state['best_x'] = xb.copy()
                except Exception:
                    pass
            else:
                from scipy.optimize import minimize
                try:
                    res = minimize(
                        lambda x: call(x),
                        best.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                 'adaptive': True},
                    )
                    if np.isfinite(res.fun) and res.fun < state['best_f']:
                        state['best_f'] = float(res.fun)
                        state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
                except Exception:
                    pass

        if state['best_x'] is None:
            state['best_x'] = np.zeros(dim)
            state['best_f'] = call(state['best_x'])

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
