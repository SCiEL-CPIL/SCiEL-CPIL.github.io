"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using decomposition and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate weights on simplex for DTLZ1 (3 objectives)
        # Using random points on the 2-simplex
        def random_simplex_point(n_obj):
            u = rng.random(n_obj)
            u = u / u.sum()
            return u
        
        # Create a set of weight vectors
        n_weights = 30
        weights = np.array([random_simplex_point(self.n_obj) for _ in range(n_weights)])
        
        # Initial population: one solution per weight vector
        X = np.zeros((n_weights, self.dim))
        for i in range(n_weights):
            # For DTLZ1, the Pareto front is at x_d = 0.5 for all d
            # We can initialize near the known Pareto front
            # x_1..x_{n-2} determine the objective values, x_{n-1}..x_n should be 0.5
            X[i, :self.dim-2] = 0.5 * rng.random(self.dim-2)
            X[i, self.dim-2:] = 0.5
        
        F = np.array([problem(x) for x in X])
        used = n_weights
        
        # Local search: perturb solutions and keep if they improve their assigned weight
        # We'll do iterative local search
        max_iterations = self.budget - used
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Pick a random solution to perturb
            idx = rng.integers(n_weights)
            parent = X[idx].copy()
            
            # Random perturbation with decreasing step size
            step = 0.1 * (1.0 - used / max(1, self.budget))
            step = max(step, 0.01)
            
            child = parent + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child is better for the assigned weight
            # For DTLZ1 with weights, we want to minimize the weighted sum
            # But better to use non-domination
            # Simple replacement: replace if child is non-dominated or improves hypervolume contribution
            # For simplicity, replace if child is better in at least one objective and not worse in others
            # Or just replace if it's non-dominated by the current population
            
            # Compute if child is dominated by any existing solution
            dominated = False
            for j in range(n_weights):
                if j != idx:
                    if np.all(F[j] <= fc) and np.any(F[j] < fc):
                        dominated = True
                        break
            
            if not dominated:
                X[idx] = child
                F[idx] = fc
            else:
                # Try to find a solution that dominates the child and replace it
                # Or just keep the best solution for this weight based on weighted sum
                pass
        
        # Final: return the population
        return X
# EVOLVE-BLOCK-END
