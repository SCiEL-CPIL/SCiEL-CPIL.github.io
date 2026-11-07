"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Hybrid DE and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)
        
        # Track calls and best
        calls = 0
        best_x = None
        best_fx = np.inf

        def tracked_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution
        # Reserve 50 calls for Nelder-Mead
        de_budget = max(10, budget - 50)
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, n))
        fpop = np.array([tracked_func(x) for x in pop])
        
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_fx = fpop[best_idx]

        # Run DE until budget or time is exhausted
        while calls < de_budget:
            if time.time() > deadline:
                break
            
            i = rng.integers(pop_size)
            indices = rng.choice(pop_size, 3, replace=False)
            a, b, c = indices
            
            v = pop[a] + F * (pop[b] - pop[c])
            
            j_rand = rng.integers(n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    pass
                else:
                    v[j] = pop[i][j]
            
            v = np.clip(v, -5, 5)
            try:
                fv = tracked_func(v)
            except TimeoutError:
                break
            
            if fv <= fpop[i]:
                pop[i] = v
                fpop[i] = fv
                if fv < best_fx:
                    best_fx = fv
                    best_x = v.copy()
                    best_idx = i

        # Phase 2: Nelder-Mead Polish
        remaining_budget = budget - calls
        if remaining_budget > 0 and best_x is not None:
            # Wrap func for NM to stop if budget exceeded
            nm_calls = 0
            def nm_func(x):
                nonlocal nm_calls, calls, best_x, best_fx
                if calls >= budget or time.time() > deadline:
                    raise Exception("Budget Exceeded")
                x = np.clip(x, -5, 5)
                fx = float(func(x))
                calls += 1
                nm_calls += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                return fx

            try:
                res = scipy.optimize.minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining_budget, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                # Update best if NM found something better (tracked_func updates non-locals)
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                # If NM crashes, we keep the calls made so far
                pass

        # Ensure best_x is not None
        if best_x is None:
            best_x = np.zeros(n)
            best_fx = tracked_func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
