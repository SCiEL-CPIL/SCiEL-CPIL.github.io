# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_search(self, func, x0, budget, rng, deadline):
        """
        Run Nelder-Mead simplex optimization from x0.
        Returns (best_x, best_f, calls_made).
        """
        dim = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex
        # x0 is the first vertex
        # Other vertices are x0 + delta * e_i
        # Choose delta based on scale of domain [-5, 5]
        delta = 0.5 
        
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            v = np.clip(v, -5, 5)
            simplex.append(v)
            
        # Evaluate initial simplex
        fvals = []
        calls = 0
        for v in simplex:
            fv = func(v)
            fvals.append(fv)
            calls += 1
            if time.time() >= deadline:
                break
        
        # If we ran out of time during init, return best
        if calls < len(simplex):
            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx], calls

        best_x = simplex[np.argmin(fvals)].copy()
        best_f = min(fvals)
        
        # Main loop
        while calls < budget:
            if time.time() >= deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence: if the spread of fvals is small, stop
            if len(fvals) > 1 and (fvals[-1] - fvals[0]) < 1e-10:
                break
                
            x_w = simplex[-1]  # worst
            x_b = simplex[0]   # best
            x_c = np.mean(simplex, axis=0) # centroid of best dim points
            
            # Reflection
            x_r = x_c + alpha * (x_c - x_w)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
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
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                # Inside contraction if f_r < fvals[-1]
                if f_r < fvals[-1]:
                    x_c_new = x_c + rho * (x_w - x_c)
                    x_c_new = np.clip(x_c_new, -5, 5)
                    f_c = func(x_c_new)
                    calls += 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c_new
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            x_s = x_b + sigma * (simplex[i] - x_b)
                            x_s = np.clip(x_s, -5, 5)
                            f_s = func(x_s)
                            calls += 1
                            simplex[i] = x_s
                            fvals[i] = f_s
                else:
                    # Outside contraction
                    x_c_new = x_c - rho * (x_w - x_c)
                    x_c_new = np.clip(x_c_new, -5, 5)
                    f_c = func(x_c_new)
                    calls += 1
                    
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c_new
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            x_s = x_b + sigma * (simplex[i] - x_b)
                            x_s = np.clip(x_s, -5, 5)
                            f_s = func(x_s)
                            calls += 1
                            simplex[i] = x_s
                            fvals[i] = f_s
                            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
                
            if calls >= budget:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        # Phase 1: Multi-start Nelder-Mead
        # Use 10 diverse starting points
        n_starts = 10
        # Reserve some budget for the final polish
        polish_budget = max(20, budget // 10)
        exploration_budget = budget - polish_budget
        per_start_budget = exploration_budget // n_starts

        # Generate starts: origin, random, and some structured
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            # Mix of uniform and slightly biased towards center
            if i % 2 == 0:
                starts.append(rng.uniform(-4, 4, dim))
            else:
                starts.append(rng.uniform(-2, 2, dim))

        for s in starts:
            if call_count >= budget or time.time() >= deadline:
                break
            
            # Allocate budget for this start, but don't exceed remaining
            remaining_for_start = min(per_start_budget, budget - call_count)
            if remaining_for_start <= dim + 1: # Need at least dim+1 for simplex init
                break
                
            x, f, c = self._nm_search(func, s, remaining_for_start, rng, deadline)
            call_count += c
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Final Polish with Nelder-Mead from best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            x, f, c = self._nm_search(func, best_x, remaining, rng, deadline)
            call_count += c
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # If no valid best_x found (should not happen), fallback
        if best_x is None:
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
