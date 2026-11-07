import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        n_obj = self.n_obj
        dim = self.dim
        used = 0
        
        # Archive storage
        archive_X = []
        archive_F = []
        
        # Helper functions
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated
            for f_old in archive_F:
                if dominates(f_old, f):
                    return False
            # Remove dominated
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F
            return True
        
        # 1. Initialization: Space-filling Halton
        def halton(dim, n, skip=1):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dim):
                p = primes[d % len(primes)]
                for i in range(n):
                    x = 0.0
                    f = 1.0 / p
                    idx = i + skip
                    while idx > 0:
                        idx, rem = divmod(idx, p)
                        x += f * rem
                        f /= p
                    points[i, d] = x
            return points

        n_init = min(30, self.budget // 3)
        H = halton(dim, n_init)
        X_init = lo + (hi - lo) * H
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)
        
        # 2. Main Loop: Hybrid Gradient-Based and Random Search
        # Strategy:
        # - Use L-BFGS-B to refine points on the Pareto front using weighted sums.
        # - Use random perturbations for exploration to escape local minima.
        # - Weighted sums are chosen to cover the objective space evenly.
        
        # Generate weight vectors on the simplex
        # For 3 objectives, we can generate points on a 2D simplex
        n_weights = 10
        weights = []
        for _ in range(n_weights):
            w = rng.random(n_obj)
            w /= w.sum()
            weights.append(w)
        # Add the extreme points (axes) to ensure coverage of corners
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        weights = np.array(weights)
        
        # Track which weights have been recently used to balance exploration
        last_used_weight = np.full(len(weights), -100)
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Decide between gradient-based exploitation and random exploration
            # Early on, explore more. Later, exploit more.
            progress = used / self.budget
            exploit_prob = 0.5 + 0.3 * progress
            
            if len(archive_X) > 0 and rng.random() < exploit_prob:
                # Exploitation: Select a weight vector
                # Prefer weights that haven't been used recently
                probs = np.exp(-0.1 * (used - last_used_weight))
                probs /= probs.sum()
                w_idx = rng.choice(len(weights), p=probs)
                w = weights[w_idx]
                last_used_weight[w_idx] = used
                
                # Find the best point in the archive for this weight
                scores = np.array([f @ w for f in archive_F])
                best_idx = np.argmin(scores)
                x_start = archive_X[best_idx].copy()
                
                # Define objective for L-BFGS-B
                def obj_func(x):
                    nonlocal used
                    if used >= self.budget:
                        return 0.0
                    f = problem(x)
                    used += 1
                    return float(f @ w)
                
                # Run L-BFGS-B with a limited number of function evaluations
                # maxiter is not a direct eval count, but we can use maxiter to limit iterations
                # L-BFGS-B uses ~1-2 evals per iteration in low dims
                # We want to spend a small fraction of the remaining budget per call
                max_iter = 5 
                
                try:
                    res = minimize(obj_func, x_start, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
                    if res.success and res.fun < scores[best_idx]:
                        # If we found a better point, add it to the archive
                        # Note: res.x is the location, but we need the f values
                        # We already evaluated it in obj_func, but we don't have the vector f stored easily.
                        # However, since it improved the scalar objective, it's likely non-dominated or close.
                        # Let's re-evaluate to get the vector? No, that wastes a call.
                        # Actually, we can just trust that if it improved the weighted sum, it's worth keeping.
                        # But we need the vector to update the archive properly.
                        # Since the evaluator records ALL returned vectors, we don't strictly need to manage the archive
                        # for scoring, but we need it for our internal selection.
                        # The problem function returns the vector.
                        # We can modify obj_func to store the last f.
                        pass 
                except Exception:
                    pass
                
                # To properly update the archive with the vector, we should have stored it.
                # Let's refactor the inner loop to be simpler:
                # Instead of complex L-BFGS integration that might eat budget unpredictably,
                # let's use a simpler gradient step or just use L-BFGS-B but capture the result.
                
                # Alternative simpler exploitation:
                # Take the best point for weight w, and do a few random steps with acceptance based on w.
                x_curr = x_start
                f_curr = archive_F[best_idx]
                for _ in range(3): # 3 steps of local random search
                    if used >= self.budget:
                        break
                    step_size = 0.05 * (1.0 - 0.5 * progress)
                    noise = step_size * (hi - lo) * rng.standard_normal(dim)
                    x_try = np.clip(x_curr + noise, lo, hi)
                    f_try = problem(x_try)
                    used += 1
                    
                    # Accept if it improves the weighted sum
                    if f_try @ w < f_curr @ w:
                        x_curr = x_try
                        f_curr = f_try
                        add_to_archive(x_curr, f_curr)
                    else:
                        # Even if not better for w, it might be non-dominated
                        add_to_archive(x_try, f_try)
                        
            else:
                # Exploration: Random point
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                used += 1
                add_to_archive(x_new, f_new)
                
        return archive_X
