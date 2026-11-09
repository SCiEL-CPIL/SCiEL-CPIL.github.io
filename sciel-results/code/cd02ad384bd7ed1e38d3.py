import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        n_evals = 0
        
        # Helper to evaluate and track budget
        def evaluate(x):
            nonlocal n_evals, best_x, best_f
            if n_evals >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            n_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead implementation
        def nelder_mead(func_eval, x0, max_iter, initial_step=0.5):
            # Initialize simplex
            simplex = [x0.copy()]
            f_vals = [func_eval(x0)]
            
            for i in range(dim):
                if n_evals >= budget:
                    return min(f_vals) if f_vals else float('inf')
                x_new = x0.copy()
                x_new[i] += initial_step
                simplex.append(x_new)
                f_vals.append(func_eval(x_new))
            
            # If we ran out of budget during init, return
            if len(f_vals) < dim + 1:
                return min(f_vals)
                
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                # Check budget for potential new evaluations
                if n_evals >= budget - 5: 
                    break

                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                
                # Centroid of best points
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = func_eval(x_ref)
                if n_evals >= budget:
                    break
                    
                if f_ref < f_vals[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_worst)
                    f_exp = func_eval(x_exp)
                    if n_evals >= budget:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    f_con = func_eval(x_con)
                    if n_evals >= budget:
                        break
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func_eval(simplex[i])
                            if n_evals >= budget:
                                break
                        if n_evals >= budget:
                            break

            return min(f_vals) if f_vals else float('inf')

        # Multi-start strategy
        n_restarts = 5
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
            
            if r == 0:
                x_start = np.zeros(dim)
            else:
                # Deterministic pseudo-random restarts based on restart index
                # to ensure coverage
                np.random.seed(1234 + r)
                x_start = np.random.uniform(-4, 4, dim)
            
            # Determine max iterations for this restart
            remaining = budget - n_evals
            # Reserve some budget for other restarts if not the last one
            if r < n_restarts - 1:
                # Use a portion of the remaining budget
                max_iter = max(10, (remaining // (n_restarts - r)) // 2)
            else:
                max_iter = max(10, remaining // 2)
            
            f_res = nelder_mead(evaluate, x_start, max_iter)
            
            if n_evals >= budget:
                break

        if best_x is None:
            # Fallback if no evaluation was made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            
        return best_x, best_f
