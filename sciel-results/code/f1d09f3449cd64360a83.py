# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(mu+lambda)-ES with adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 325.0

        rng = np.random.default_rng(42)

        mu = 2
        lam = 8

        # Initialize population
        pop_x = rng.uniform(-5, 5, (mu, n))
        pop_f = np.array([func(x) for x in pop_x])
        
        best_x = pop_x[np.argmin(pop_f)]
        best_f = np.min(pop_f)
        eval_count = mu

        sigma = 1.0
        sigma_min = 1e-8
        sigma_max = 50.0

        # Sort population by fitness
        idx = np.argsort(pop_f)
        pop_x = pop_x[idx]
        pop_f = pop_f[idx]

        while eval_count < budget:
            if time.time() > deadline - 3.0:
                break

            # Check if we have enough budget for a full generation
            if budget - eval_count < lam:
                # Use remaining budget for simple (1+1) steps from best
                while eval_count < budget:
                    if time.time() > deadline - 3.0:
                        break
                    z = rng.standard_normal(n)
                    child = np.clip(pop_x[0] + sigma * z, -5, 5)
                    f_child = func(child)
                    eval_count += 1
                    if f_child < best_f:
                        best_f = f_child
                        best_x = child.copy()
                break

            # Generate offspring
            off_x = []
            off_f = []
            successes = 0
            
            off_per_parent = lam // mu
            
            for i in range(mu):
                parent = pop_x[i]
                f_parent = pop_f[i]
                for _ in range(off_per_parent):
                    z = rng.standard_normal(n)
                    child = np.clip(parent + sigma * z, -5, 5)
                    f_child = func(child)
                    eval_count += 1
                    off_x.append(child)
                    off_f.append(f_child)
                    
                    if f_child < f_parent:
                        successes += 1
                        
                    if f_child < best_f:
                        best_f = f_child
                        best_x = child.copy()

            # Combine parents and offspring
            all_x = np.vstack([pop_x, np.array(off_x)])
            all_f = np.concatenate([pop_f, np.array(off_f)])
            
            # Select best mu
            idx = np.argsort(all_f)
            pop_x = all_x[idx[:mu]]
            pop_f = all_f[idx[:mu]]
            
            # Adapt sigma
            success_rate = successes / lam
            if success_rate > 0.15:
                sigma *= 1.1
            else:
                sigma *= 0.9
                
            sigma = np.clip(sigma, sigma_min, sigma_max)

        return best_x, best_f
# EVOLVE-BLOCK-END
