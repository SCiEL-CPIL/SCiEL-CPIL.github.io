"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using adaptive grid refinement on x0
       combined with local search on remaining dimensions."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT4: x0 in [-5,5], x1..x9 in [0,1]
        # f1 = 1 + 10*(n-1) + sum((xi-0.5)^2 - 10*cos(2*pi*(xi-0.5))) for i=2..n
        # f2 = (f1+1) * (1 - sqrt(f1/(f1+1)))
        # f1 is minimized when all xi=0.5, giving f1 = 1
        # The Pareto front is parameterized by x0 which controls the trade-off
        
        # Strategy: adaptive grid refinement on x0
        # 1. Initial uniform sampling of x0
        # 2. For each x0, find optimal x1..x9 (which should be near 0.5)
        # 3. Refine x0 intervals with best hypervolume contribution
        
        # Initial grid of x0 values
        n_init = 30
        x0_grid = np.linspace(lo[0], hi[0], n_init)
        
        # For each x0, we need to optimize x1..x9
        # Since f1 is separable in x1..x9 and each term is minimized at 0.5,
        # the optimal x1..x9 are all 0.5
        # But let's do a local search to be safe
        
        def optimize_remaining(x0, x_init=None):
            """Optimize x1..x9 for a fixed x0 using local search."""
            if x_init is None:
                x_rem = np.full(dim - 1, 0.5)
            else:
                x_rem = x_init.copy()
            
            # Local search with decreasing step size
            step = 0.1
            for _ in range(10):
                best_x = x_rem.copy()
                best_f = np.inf
                for i in range(dim - 1):
                    for delta in [-step, step]:
                        candidate = best_x.copy()
                        candidate[i] += delta
                        candidate = np.clip(candidate, lo[1:], hi[1:])
                        x_full = np.concatenate(([x0], candidate))
                        f = problem(x_full)
                        f1 = f[0]
                        if f1 < best_f:
                            best_f = f1
                            best_x = candidate
                x_rem = best_x
                step *= 0.7
            
            return x_rem
        
        # Evaluate initial grid
        solutions = []  # list of (x, f)
        used = 0
        
        for x0 in x0_grid:
            x_rem = optimize_remaining(x0)
            x = np.concatenate(([x0], x_rem))
            f = problem(x)
            solutions.append((x, f))
            used += 1
            
            if used >= budget or time.time() > deadline:
                break
        
        # Adaptive refinement: identify intervals with best hypervolume contribution
        # Sort solutions by f1
        solutions.sort(key=lambda sf: sf[1][0])
        
        # Refinement loop
        n_refine = 50
        refine_budget = min(n_refine, budget - used)
        
        # Create intervals from sorted solutions
        # For each pair of adjacent solutions, estimate hypervolume contribution
        # and refine the interval with the largest contribution
        
        for _ in range(refine_budget):
            if time.time() > deadline:
                break
            
            # Find the interval with the largest gap in f1 space
            # among non-dominated solutions
            if len(solutions) < 2:
                break
            
            # Get unique f1 values
            f1_vals = np.array([sf[1][0] for sf in solutions])
            x0_vals = np.array([sf[0][0] for sf in solutions])
            
            # Find the largest gap between consecutive f1 values
            if len(f1_vals) >= 2:
                diffs = np.diff(f1_vals)
                max_idx = np.argmax(diffs)
                # Insert a new x0 in the middle of this gap
                x0_new = (x0_vals[max_idx] + x0_vals[max_idx + 1]) / 2
            else:
                x0_new = (lo[0] + hi[0]) / 2
            
            x_rem = optimize_remaining(x0_new)
            x = np.concatenate(([x0_new], x_rem))
            f = problem(x)
            solutions.append((x, f))
            used += 1
        
        # Final local polish on the best solutions
        # Take the top 10 solutions by hypervolume contribution and polish them
        if used < budget and time.time() < deadline:
            # Sort by f1 and take a diverse set
            solutions.sort(key=lambda sf: sf[1][0])
            
            # Take every k-th solution to get diversity
            step_size = max(1, len(solutions) // 10)
            selected = solutions[::step_size][:10]
            
            for i, (x, f) in enumerate(selected):
                if used >= budget or time.time() > deadline:
                    break
                
                # Polish x1..x9 with smaller steps
                x0 = x[0]
                x_rem = x[1:].copy()
                step = 0.05
                for _ in range(5):
                    best_x = x_rem.copy()
                    best_f = np.inf
                    for j in range(dim - 1):
                        for delta in [-step, step]:
                            candidate = best_x.copy()
                            candidate[j] += delta
                            candidate = np.clip(candidate, lo[1:], hi[1:])
                            x_full = np.concatenate(([x0], candidate))
                            f_val = problem(x_full)
                            f1_val = f_val[0]
                            if f1_val < best_f:
                                best_f = f1_val
                                best_x = candidate
                    x_rem = best_x
                    step *= 0.5
                
                x_new = np.concatenate(([x0], x_rem))
                f_new = problem(x_new)
                solutions.append((x_new, f_new))
                used += 1
        
        return solutions
# EVOLVE-BLOCK-END
