"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid approach combining multi-start simulated annealing for global exploration
    and Nelder-Mead for local refinement on f22 (Gallagher 21 peaks).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Quasi-random scan to find promising basins
        # Use Halton sequence for better coverage
        def halton_sequence(index, dim):
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        # Scan phase: 30 points to leave more budget for SA + NM
        n_scan = min(30, budget // 15)
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Phase 2: Multi-start Simulated Annealing
        # This exploits multimodality better than CMA-ES for f22
        def simulated_annealing(x_start, max_evals):
            nonlocal calls, best_f, best_x
            
            if calls >= budget or time.time() > deadline:
                return
            
            x_curr = x_start.copy()
            f_curr = eval_and_track(x_curr)
            
            # SA parameters
            T_start = 5.0
            T_end = 0.01
            cooling = 0.95
            step_size = 1.0
            
            T = T_start
            evals_sa = 0
            
            while T > T_end and evals_sa < max_evals:
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate perturbation
                perturbation = rng.standard_normal(dim) * step_size
                x_trial = np.clip(x_curr + perturbation, -self.bound, self.bound)
                f_trial = eval_and_track(x_trial)
                evals_sa += 1
                
                # Accept or reject
                delta_f = f_trial - f_curr
                if delta_f < 0 or rng.random() < np.exp(-delta_f / T):
                    x_curr = x_trial
                    f_curr = f_trial
                    # Gradually decrease step size
                    step_size *= 0.99
                else:
                    step_size = min(step_size * 1.05, 3.0)
                
                # Cool down
                T *= cooling
            
            return x_curr, f_curr

        # Run multi-start SA
        if best_f < float('inf') and calls < budget - 50:
            remaining = budget - calls
            # Allocate 70% of remaining budget to SA
            sa_budget = int(remaining * 0.7)
            if sa_budget > 20:
                # Multiple restarts from different starting points
                n_restarts = min(5, sa_budget // 20)
                sa_per_restart = sa_budget // n_restarts
                
                for r in range(n_restarts):
                    if calls >= budget - 50 or time.time() > deadline:
                        break
                    
                    # Choose starting point: best so far or random
                    if r == 0:
                        x_start = best_x.copy()
                    else:
                        x_start = rng.uniform(-4, 4, size=dim)
                    
                    simulated_annealing(x_start, sa_per_restart)
        
        # Phase 3: Local refinement with Nelder-Mead
        if calls < budget - 10:
            remaining = budget - calls
            # Use Nelder-Mead on the best point
            x_start = best_x.copy()
            
            class BudgetFunc:
                def __init__(self, func, x_best, f_best, budget_remaining):
                    self.func = func
                    self.x_best = x_best
                    self.f_best = f_best
                    self.remaining = budget_remaining
                    self.count = 0
                
                def __call__(self, x):
                    if self.count >= self.remaining or time.time() > deadline:
                        return self.f_best
                    x_clipped = np.clip(x, -self.bound, self.bound)
                    fx = self.func(x_clipped)
                    self.count += 1
                    if fx < self.f_best:
                        self.f_best = fx
                        self.x_best = x_clipped.copy()
                    return fx
            
            bf = BudgetFunc(func, x_start, best_f, remaining)
            
            try:
                res = minimize(bf, x_start, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # Phase 4: Final coordinate descent polish
        if calls < budget - 5:
            # Simple coordinate descent with adaptive step
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.5
            for _ in range(15):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    # Try moving in both directions
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = eval_and_track(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            # Try to continue in same direction
                            for _cont in range(3):
                                if calls >= budget - 5 or time.time() > deadline:
                                    break
                                x_trial2 = x_curr.copy()
                                x_trial2[d] += direction * step
                                x_trial2 = np.clip(x_trial2, -5, 5)
                                f_trial2 = eval_and_track(x_trial2)
                                if f_trial2 < f_curr:
                                    x_curr = x_trial2
                                    f_curr = f_trial2
                                else:
                                    break
                    
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break
                else:
                    step = max(step * 1.1, 0.1)
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr

        return best_x, best_f

# EVOLVE-BLOCK-END
