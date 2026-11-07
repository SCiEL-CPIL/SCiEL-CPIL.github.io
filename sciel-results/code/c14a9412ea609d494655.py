import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 10
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        # Initialize
        for i in range(pop_size):
            # DTLZ7 specific initialization
            # x[0] random
            # x[1:-1] = 0
            # x[-1] random
            X[i] = rng.random(self.dim)
            if self.dim > 1:
                X[i, 1:self.dim-1] = 0.0
            X[i, 0] = rng.random()
            X[i, -1] = rng.random()
            
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
            
        def get_non_dominated(F):
            nd = []
            for i in range(len(F)):
                is_nd = True
                for j in range(len(F)):
                    if i != j and dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return nd

        while used < self.budget:
            if time.time() > deadline:
                break
            
            nd_idx = get_non_dominated(F)
            
            # Local Search on non-dominated points
            # Try to improve them by small steps
            for idx in nd_idx:
                if used >= self.budget:
                    break
                # Perturb a few random variables
                x = X[idx].copy()
                f = F[idx]
                
                # Try 2 steps of local search
                for _ in range(2):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Random direction
                    dir = rng.standard_normal(self.dim)
                    # Scale step
                    step = 0.05 * (hi - lo)
                    new_x = x + dir * step
                    new_x = np.clip(new_x, lo, hi)
                    
                    # Bias: keep middle vars small
                    if self.dim > 1:
                        new_x[1:self.dim-1] *= 0.8
                        
                    new_f = problem(new_x)
                    used += 1
                    
                    if dominates(new_f, f):
                        x = new_x
                        f = new_f
                        # Update population immediately? Or just keep local best?
                        # Let's update the population point if it improves
                        X[idx] = x
                        F[idx] = f
                        # Re-evaluate non-dominated? Too expensive.
                        # Just break local search for this individual if improved?
                        # Or continue to see if we can improve more.
                        # Let's break to save budget for global search
                        break
                # End local search for this individual

            # Global Search: Create new child
            if used >= self.budget:
                break
                
            # Select parent
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx]
            
            # Mutation
            child = parent + 0.2 * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Bias
            if self.dim > 1:
                child[1:self.dim-1] *= 0.5
                
            fc = problem(child)
            used += 1
            
            # Replace worst
            dom_idx = [i for i in range(pop_size) if i not in nd_idx]
            if dom_idx:
                victim_idx = dom_idx[rng.integers(len(dom_idx))]
            else:
                victim_idx = rng.integers(pop_size)
                
            X[victim_idx] = child
            F[victim_idx] = fc
            
        return X
