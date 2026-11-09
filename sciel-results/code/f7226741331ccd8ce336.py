import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        n_evals = 0
        X = []
        F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        # 1. Latin Hypercube Sampling for initial coverage
        # Use 100 points for global exploration
        n_lhs = 100
        if n_lhs > self.budget:
            n_lhs = self.budget
            
        # Generate LHS points
        # For each dim, create n_lhs unique intervals
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Create bins
            edges = np.linspace(0, 1, n_lhs + 1)
            # Random point in each bin
            points_in_bins = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle to decorrelate dimensions
            np.random.shuffle(points_in_bins)
            lhs_points[:, d] = points_in_bins
        
        # Scale to bounds
        lhs_points = lo + lhs_points * span
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = lhs_points[i]
            fx = evaluate(x)
            X.append(x)
            F.append(fx)
            
        if n_evals >= self.budget or time.time() > deadline:
            return np.array(X)
            
        # 2. Local Search around best points
        # Convert to arrays for easy manipulation
        X = np.array(X)
        F = np.array(F)
        
        # Find non-dominated points
        # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
        def is_dominated(F, i):
            for j in range(len(F)):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    return True
            return False
            
        nondom_indices = [i for i in range(len(F)) if not is_dominated(F, i)]
        
        if not nondom_indices:
            # Fallback: pick random
            nondom_indices = list(range(len(F)))
            
        # Select top K non-dominated points for local search
        # Sort by sum of objectives (simple heuristic)
        nondom_indices.sort(key=lambda i: np.sum(F[i]))
        k_parents = min(10, len(nondom_indices))
        parents = [X[i] for i in nondom_indices[:k_parents]]
        
        # Use remaining budget for local search
        # Distribute remaining evaluations among parents
        remaining_budget = self.budget - n_evals
        if remaining_budget > 0 and parents:
            n_children_per_parent = remaining_budget // len(parents)
            if n_children_per_parent == 0:
                n_children_per_parent = 1
                # If we can't afford 1 per parent, just do what we can
            
            # We will iterate parents and generate children
            # To ensure we use the budget exactly, we might need a loop
            # But simple loop:
            child_idx = 0
            while n_evals < self.budget and time.time() < deadline:
                parent = parents[child_idx % len(parents)]
                # Gaussian perturbation
                # Scale step size to 10% of range
                step = 0.1 * span
                noise = np.random.standard_normal(self.dim)
                child = parent + noise * step
                child = np.clip(child, lo, hi)
                
                fc = evaluate(child)
                X = np.vstack([X, child])
                F = np.vstack([F, fc])
                child_idx += 1
                
                # Optional: If child is better than parent, update parent?
                # For simplicity and speed, we just record.
                
        return np.array(X)
