"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start and adaptive re-seeding."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        def nelder_mead(func, x0, max_calls, sigma_init=1.0):
            """Run Nelder-Mead from x0 with at most max_calls evaluations."""
            nonlocal calls_made
            dim = self.dim
            x0 = np.clip(x0, -5, 5)
            
            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [evaluate(x0)]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += sigma_init
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(evaluate(xi))
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_shrink = 0.5
            
            calls_left = max_calls - (calls_made - (max_calls - max_calls))  # just track by counting
            # Actually let's just count locally
            local_calls = dim + 1  # initial simplex evaluations
            
            while local_calls < max_calls:
                # Check deadline
                if time.time() >= deadline - 5:
                    break
                
                # Current worst and best
                f_best = fvals[0]
                f_worst = fvals[-1]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)
                local_calls += 1
                
                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    local_calls += 1
                    
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = evaluate(xc)
                    local_calls += 1
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = evaluate(simplex[i])
                            local_calls += 1
                        if local_calls >= max_calls:
                            break
                
                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Stagnation check: if best hasn't improved significantly, break
                if fvals[0] > f_best * 0.999 and fvals[0] > f_best - 1e-6:
                    # Check simplex spread
                    spread = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, dim + 1)])
                    if spread < 1e-3:
                        break
            
            # Return best from this run
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Multi-start strategy
        # Phase 1: Several short Nelder-Mead runs from different starts
        # Phase 2: One longer run from the best point found
        
        # Determine how many starts based on budget
        # Each NM run needs at least dim+1 + some iterations
        # Let's do 5 starts with 80 calls each, then 1 final run with remaining budget
        n_starts = 5
        calls_per_start = max(30, (budget - 150) // n_starts)
        final_budget = budget - n_starts * calls_per_start
        if final_budget < 20:
            final_budget = 50
            calls_per_start = max(20, (budget - final_budget) // n_starts)

        # Initial diverse starts
        starts = []
        # One at center
        starts.append(np.zeros(dim))
        # Random starts
        for _ in range(n_starts - 1):
            x = rng.uniform(-4, 4, dim)
            starts.append(x)
        
        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        for i, x0 in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() >= deadline - 10:
                break
            
            remaining = budget - calls_made
            alloc = min(calls_per_start, remaining)
            if alloc < dim + 2:
                break
            
            x_best, f_best_local = nelder_mead(func, x0, alloc)
            
            if f_best_local < best_f:
                best_f = f_best_local
                best_x = x_best.copy()
        
        # Final polish run from best point
        if best_x is not None and calls_made < budget:
            if time.time() >= deadline - 10:
                return best_x, best_f
            
            remaining = budget - calls_made
            if remaining > dim + 2:
                x_final, f_final = nelder_mead(func, best_x, remaining)
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final.copy()
        
        if best_x is None:
            # Fallback: evaluate at origin
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
