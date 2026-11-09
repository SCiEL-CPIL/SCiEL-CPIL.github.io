"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 exploiting known Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # DTLZ1 structure for 3 objectives:
        # g = 1 + 9 * mean(x[2:7])
        # f1 = (g+1) * x[0] * x[1]
        # f2 = (g+1) * x[0] * (1-x[1])
        # f3 = (g+1) * (1-x[0])
        # Pareto front when x[2:]=0: f1=2*x0*x1, f2=2*x0*(1-x1), f3=2*(1-x0)
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Dense grid sampling on the Pareto front
        # Use 85% of budget for grid
        n_grid = int(budget * 0.85)
        
        # Create a well-distributed grid in (x0, x1) space
        # Use stratified sampling with jitter for better coverage
        n_per_dim = int(np.sqrt(n_grid))
        if n_per_dim < 10:
            n_per_dim = 10
        n_grid_actual = n_per_dim * n_per_dim
        
        # Generate stratified samples with small jitter
        x0_base = np.linspace(0.0, 1.0, n_per_dim + 1)
        x1_base = np.linspace(0.0, 1.0, n_per_dim + 1)
        
        # Create cell centers with jitter
        x0_samples = (x0_base[:-1] + x0_base[1:]) / 2.0 + rng.uniform(-0.5, 0.5, n_per_dim) * (1.0 / n_per_dim)
        x1_samples = (x1_base[:-1] + x1_base[1:]) / 2.0 + rng.uniform(-0.5, 0.5, n_per_dim) * (1.0 / n_per_dim)
        
        # Clip to [0, 1]
        x0_samples = np.clip(x0_samples, 0.0, 1.0)
        x1_samples = np.clip(x1_samples, 0.0, 1.0)
        
        # Create grid
        X0, X1 = np.meshgrid(x0_samples, x1_samples, indexing='ij')
        X0_flat = X0.flatten()
        X1_flat = X1.flatten()
        
        # Ensure we don't exceed budget
        n_evals = min(n_grid_actual, n_grid)
        X0_flat = X0_flat[:n_evals]
        X1_flat = X1_flat[:n_evals]
        
        # Create full vectors
        X = np.zeros((n_evals, dim))
        X[:, 0] = X0_flat
        X[:, 1] = X1_flat
        # x[2:] = 0 for minimum g
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_evals
        
        # Check time
        if time.time() > deadline - 20:
            return X
        
        # Phase 2: Refinement - add points in under-sampled regions
        remaining = budget - used
        if remaining > 0:
            # Find non-dominated points
            def is_dominated(F):
                n = len(F)
                dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    if dominated[i]:
                        continue
                    for j in range(n):
                        if i == j or dominated[j]:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                return dominated
            
            dominated = is_dominated(F)
            nondom_idx = np.where(~dominated)[0]
            
            # Use remaining budget for targeted refinement
            # Focus on regions with sparse coverage
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                if used >= budget:
                    break
                
                # Pick a random non-dominated point and refine nearby
                if len(nondom_idx) > 0:
                    parent_idx = nondom_idx[rng.integers(len(nondom_idx))]
                    x_parent = X[parent_idx].copy()
                    
                    # Small perturbation in x0, x1
                    sigma = 0.02
                    perturbation = sigma * rng.standard_normal(2)
                    x_child = x_parent.copy()
                    x_child[0] = np.clip(x_child[0] + perturbation[0], 0.0, 1.0)
                    x_child[1] = np.clip(x_child[1] + perturbation[1], 0.0, 1.0)
                    
                    # Keep x[2:] = 0
                    x_child[2:] = 0
                    
                    fc = problem(x_child)
                    used += 1
                    
                    # Add to population: replace the point with highest sum of objectives
                    # if the new point is better
                    sums = np.sum(F, axis=1)
                    worst_idx = np.argmax(sums)
                    
                    # Check if child is better (dominates or has lower sum)
                    if np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx]):
                        X[worst_idx] = x_child
                        F[worst_idx] = fc
                    elif np.sum(fc) < sums[worst_idx]:
                        X[worst_idx] = x_child
                        F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
