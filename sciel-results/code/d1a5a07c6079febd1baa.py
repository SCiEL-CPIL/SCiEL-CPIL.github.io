import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj
            self.start_time = None

        def __call__(self, problem, bounds):
            self.start_time = time.time()
            deadline = self.start_time + 300.0 # Safety margin
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # x_1 is index 0, x_2..x_10 are 1..9
            # We want to minimize g(x) which depends on x_1..x_9 (indices 1..9)
            # g(x) is proportional to f_2(x) for fixed x_1?
            # f_2 = g(x) * (1 - sqrt(x_1/g(x)))
            # Minimizing g(x) is the main goal for the "shape" of the front.
            
            # Phase 1: Find good x_2..x_10
            # Budget allocation: 120 for Phase 1, 80 for Phase 2
            b1 = 120
            b2 = self.budget - b1
            
            # Initialize population for x_2..x_10
            n_init = 20
            X_sub = np.random.uniform(lo[1:], hi[1:], size=(n_init, self.dim - 1))
            
            # Evaluate g(x) for these points
            # Use x_1 = 0.5 for evaluation
            x1_eval = 0.5
            g_vals = np.zeros(n_init)
            for i in range(n_init):
                x = np.concatenate(([x1_eval], X_sub[i]))
                x = np.clip(x, lo, hi)
                f = problem(x)
                # g(x) can be recovered?
                # f_2 = g(1 - sqrt(x_1/g))
                # It's easier to just minimize f_2?
                # Minimizing f_2 for fixed x_1 is equivalent to minimizing g(x) if x_1 is fixed.
                g_vals[i] = f[1]
                
            # Keep best 5
            idx = np.argsort(g_vals)[:5]
            best_sub = X_sub[idx].copy()
            best_g = g_vals[idx].copy()
            
            used = n_init
            
            # Local search on the best 5 using Nelder-Mead
            # Each Nelder-Mead run might use ~10-20 evals
            # We have b1 - n_init = 100 evals left for Phase 1
            # Let's run Nelder-Mead on the single best point for ~20 iterations?
            # Or run on 2 points with 10 iterations each?
            
            # Define objective for minimization
            def obj_func(sub_x):
                x = np.concatenate(([x1_eval], sub_x))
                x = np.clip(x, lo, hi)
                f = problem(x)
                return f[1]
                
            # Run Nelder-Mead
            # x0 is best_sub[0]
            # maxiter should be small to save budget
            # Each iteration of NM uses N+1 = 10 function calls? No, it uses a simplex.
            # Let's just do a simple manual local search to control budget strictly.
            
            # Simple local search:
            # For each of the best 5 points, try small perturbations.
            remaining_p1 = b1 - used
            n_local = 5
            evals_per_local = remaining_p1 // n_local
            
            for i in range(n_local):
                current_sub = best_sub[i]
                current_g = best_g[i]
                for _ in range(evals_per_local):
                    if time.time() > deadline - 10: break
                    # Random perturbation
                    step = 0.5 * (hi[1:] - lo[1:])
                    perturbation = np.random.normal(0, 0.1, size=self.dim - 1) * step
                    new_sub = current_sub + perturbation
                    new_sub = np.clip(new_sub, lo[1:], hi[1:])
                    
                    x = np.concatenate(([x1_eval], new_sub))
                    f = problem(x)
                    new_g = f[1]
                    used += 1
                    
                    if new_g < current_g:
                        current_sub = new_sub
                        current_g = new_g
                        best_sub[i] = current_sub
                        best_g[i] = current_g
                        
            # Find the overall best x_2..x_10
            overall_best_idx = np.argmin(best_g)
            best_x_sub = best_sub[overall_best_idx]
            
            # Phase 2: Sample the Pareto Front
            # We have best_x_sub. Now vary x_1.
            # Start with a few points
            x1_points = [0.0, 0.5, 1.0]
            F_points = []
            
            for x1 in x1_points:
                x = np.concatenate(([x1], best_x_sub))
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_points.append(f)
                used += 1
                
            # Adaptive bisection for remaining budget
            remaining_p2 = b2 - len(x1_points)
            
            # Maintain list of (x1, f1, f2)
            # We want to maximize hypervolume.
            # Approximation: Fill largest gaps in x1.
            
            evaluated_x1s = sorted(x1_points)
            
            for _ in range(remaining_p2):
                if time.time() > deadline - 5: break
                if len(evaluated_x1s) < 2:
                    break
                    
                # Find largest gap
                max_gap = -1
                mid_x1 = 0.5
                for i in range(len(evaluated_x1s) - 1):
                    gap = evaluated_x1s[i+1] - evaluated_x1s[i]
                    if gap > max_gap:
                        max_gap = gap
                        mid_x1 = (evaluated_x1s[i] + evaluated_x1s[i+1]) / 2
                        
                x = np.concatenate(([mid_x1], best_x_sub))
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_points.append(f)
                evaluated_x1s.append(mid_x1)
                evaluated_x1s.sort()
                used += 1
                
            return np.array(F_points)
