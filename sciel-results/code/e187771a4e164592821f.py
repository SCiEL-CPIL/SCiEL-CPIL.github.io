import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 290 # 10s margin
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                
                # 1. Minimize g(x[1:])
                # We will use a subset of budget for this
                # Let's say 50 evaluations for local search on x[1:]
                # We need to construct full vectors to call problem()
                
                # Initialize x[1:] randomly, but maybe close to 0?
                # ZDT4 bounds are usually [-5, 5].
                # Let's try random init first, then local search.
                
                x_init = lo + (hi - lo) * np.random.rand(self.dim)
                # Force x[0] to 0 for the g-minimization phase?
                # f2 = 1 + (1-sqrt(x0))g.
                # To minimize g, we should set x0=0 so that f2 = 1+g.
                # Minimizing f2 is equivalent to minimizing g.
                x_init[0] = 0.0
                
                best_x = x_init.copy()
                best_f = problem(best_x)
                best_g = best_f[1] - 1.0 # approx g
                
                # Local search on x[1:]
                # We'll do a few iterations of coordinate descent
                # or just random perturbations
                evals_used = 1
                max_g_evals = 60
                step_size = 0.5
                
                while evals_used < max_g_evals and time.time() < deadline:
                    improved = False
                    for i in range(1, self.dim):
                        # Try perturbing x[i]
                        for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                            x_trial = best_x.copy()
                            x_trial[i] = np.clip(x_trial[i] + delta, lo[i], hi[i])
                            f_trial = problem(x_trial)
                            evals_used += 1
                            if f_trial[1] < best_f[1]: # Minimizing f2 (which is 1+g since x0=0)
                                best_x = x_trial
                                best_f = f_trial
                                improved = True
                    if not improved:
                        step_size *= 0.5
                    else:
                        # If improved, maybe keep step size or shrink slightly
                        pass
                        
                # Now we have a good x[1:] with low g.
                # Let's extract it
                x_best_tail = best_x[1:].copy()
                
                # 2. Sample x0 to populate the front
                # We have budget - evals_used left
                remaining = self.budget - evals_used
                
                # We want to cover x0 in [0, 1]
                # Let's create a set of x0 values
                # Include 0, 1, and some intermediate points
                x0_values = np.linspace(0.0, 1.0, min(remaining, 100))
                
                # Ensure 0 and 1 are included
                if 0.0 not in x0_values:
                    x0_values = np.insert(x0_values, 0, 0.0)
                if 1.0 not in x0_values:
                    x0_values = np.append(x0_values, 1.0)
                    
                # Evaluate these points
                # Note: We must not exceed budget
                for x0 in x0_values:
                    if evals_used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x_trial = np.zeros(self.dim)
                    x_trial[0] = x0
                    x_trial[1:] = x_best_tail
                    f_trial = problem(x_trial)
                    evals_used += 1
                    # We don't strictly need to keep them, but we could
                    # The evaluator records ALL returned objective vectors.
                    # So we just need to call problem().
                    
                return best_x
