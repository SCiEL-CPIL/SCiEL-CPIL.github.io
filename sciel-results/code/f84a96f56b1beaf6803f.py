"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time-bounded budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_remaining, deadline, n):
        """Run Nelder-Mead simplex from x0, respecting budget and deadline."""
        # Initialize simplex
        simplex = [np.array(x0, dtype=float)]
        fx0 = func(simplex[0])
        
        # Create other vertices with small offset
        for i in range(n):
            v = np.array(x0, dtype=float)
            v[i] += 0.5
            simplex.append(v)
        
        # Evaluate all vertices
        fvals = [fx0]
        for i in range(1, n + 1):
            fv = func(simplex[i])
            fvals.append(fv)
        
        calls = n + 1
        best_x = simplex[0]
        best_f = fx0
        
        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        while calls < budget_remaining and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            best_x = simplex[0]
            best_f = fvals[0]
            
            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10 * (1 + abs(np.min(fvals))):
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Simple reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < worst_f:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst)
                    x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, worst_f):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if calls >= budget_remaining or time.time() >= deadline:
                        break
        
        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start random sampling
        n_starts = 10
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Include origin and some structured points
        candidates = []
        candidates.append(np.zeros(n))
        candidates.append(np.full(n, 0.0))
        for i in range(n - 2):
            candidates.append(rng.uniform(-5, 5, n))
        
        # Evaluate candidates
        for cand in candidates:
            if calls_used >= budget:
                break
            if time.time() >= deadline:
                break
            x = np.clip(cand, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x
        
        # Phase 2: Run Nelder-Mead from best point
        remaining = budget - calls_used
        if remaining > n + 1 and time.time() < deadline:
            x_opt, f_opt, calls_nm = self._nelder_mead(func, best_x, remaining, deadline, n)
            calls_used += calls_nm
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
        
        # Phase 3: If budget remains, try more local refinements
        remaining = budget - calls_used
        if remaining > n + 1 and time.time() < deadline:
            # Try a few more random restarts with local optimization
            n_extra_starts = min(5, remaining // (n + 2))
            for _ in range(n_extra_starts):
                if calls_used >= budget or time.time() >= deadline:
                    break
                x_rand = np.clip(rng.uniform(-5, 5, n), -5, 5)
                fx_rand = func(x_rand)
                calls_used += 1
                if fx_rand < best_f:
                    best_f = fx_rand
                    best_x = x_rand
                
                # Local refinement from this point if it's good
                remaining_here = budget - calls_used
                if remaining_here > n + 1 and time.time() < deadline:
                    x_local, f_local, calls_local = self._nelder_mead(
                        func, best_x, remaining_here, deadline, n
                    )
                    calls_used += calls_local
                    if f_local < best_f:
                        best_f = f_local
                        best_x = x_local
        
        return best_x, best_f
# EVOLVE-BLOCK-END
