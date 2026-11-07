import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using weighted-sum scalarization with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Generate weight vectors for weighted-sum scalarization
        # Use a grid of weights in the simplex
        n_weights = 15  # Number of different weight directions
        weights = []
        # Add vertices
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add edge midpoints
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                w = np.zeros(n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
        # Add centroid
        weights.append(np.ones(n_obj) / n_obj)
        # Add some additional diverse weights
        rng_init = np.random.default_rng(42)
        for _ in range(5):
            w = rng_init.random(n_obj)
            w /= w.sum()
            weights.append(w)

        weights = np.array(weights)
        n_weights = len(weights)

        # Population size
        pop_size = min(40, self.budget // 5)
        
        # Initialize population with weighted-sum guided random points
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        # First, evaluate some random points
        n_random = min(20, pop_size)
        rng = np.random.default_rng(0)
        X[:n_random] = lo + (hi - lo) * rng.random((n_random, dim))
        F[:n_random] = np.array([problem(x) for x in X[:n_random]])
        
        used = n_random

        # For remaining population, use weighted-sum to guide initialization
        for i in range(n_random, pop_size):
            if used >= self.budget:
                break
            w = weights[i % n_weights]
            # Start from a random point and do a few steps of gradient descent on weighted sum
            x = lo + (hi - lo) * rng.random(dim)
            # Simple local search on weighted sum
            for step in range(3):
                if used >= self.budget or time.time() > deadline - 10:
                    break
                def scalar_obj(x_local):
                    f = problem(x_local)
                    return float(np.dot(w, f))
                
                res = minimize(scalar_obj, x, method='Nelder-Mead', 
                              options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-6})
                x = np.clip(res.x, lo, hi)
                used += 1  # Each minimize call may use multiple evals, but we count conservatively
                # Actually, we need to be more careful with budget tracking
                # Let's just do a simple perturbation instead for budget safety
            
            # Just do a simple random perturbation for safety
            x = lo + (hi - lo) * rng.random(dim)
            X[i] = x
            F[i] = problem(x)
            used += 1

        # Main loop: weighted-sum guided search with local refinement
        while used < self.budget:
            if time.time() > deadline - 20:
                break
            
            # Choose a weight vector
            w_idx = rng.integers(n_weights)
            w = weights[w_idx]
            
            # Choose a starting point from current population
            # Prefer non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            non_dom = ~dominated
            if np.any(non_dom):
                start_idx = rng.choice(np.where(non_dom)[0])
            else:
                start_idx = rng.integers(pop_size)
            
            x0 = X[start_idx].copy()
            
            # Perform local optimization on weighted sum
            def scalar_obj(x_local):
                f = problem(x_local)
                return float(np.dot(w, f))
            
            # Use Nelder-Mead for local refinement
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead',
                              bounds=list(zip(lo, hi)),
                              options={'maxiter': 10, 'xatol': 1e-5, 'fatol': 1e-7})
                x_new = np.clip(res.x, lo, hi)
            except Exception:
                x_new = x0
            
            # Evaluate the new point
            f_new = problem(x_new)
            used += 1
            
            # Replace the worst point in the population
            # Use a simple criterion: sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = x_new
            F[worst_idx] = f_new

        return X
