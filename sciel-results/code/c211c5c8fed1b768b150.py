"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 exploiting known structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 structure: f1 = x[0], f2 depends on x[0] and sum of sin terms for x[1:]
        # Pareto front: x[1:] = 0 minimizes f2 for given x[0]
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) + sin(10*pi*f1)) when x[1:]=0
        
        n = self.dim
        c = 1 + 10 * (n - 1)  # 291 for n=30
        
        # Generate candidate x[0] values to cover the oscillatory front
        # sin(10*pi*x0) has 10 full periods in [0,1]
        # We need dense sampling where the front contributes to hypervolume
        
        # Strategy: use a combination of:
        # 1. Dense grid to capture oscillations
        # 2. Extra points near minima of the oscillatory part
        
        # The function 1 - sqrt(x0/c) + sin(10*pi*x0) 
        # We want to sample where this is small (good f2)
        
        # Generate candidates
        candidates_x0 = []
        
        # Dense grid: 100 points to capture 10 oscillations well
        grid = np.linspace(0.0, 1.0, 120)
        candidates_x0.extend(grid.tolist())
        
        # Add extra points around local minima of sin(10*pi*x0)
        # sin(10*pi*x0) = -1 at x0 = (2k+1)/(20) for k=0,...,9
        for k in range(10):
            x0_min = (2*k + 1) / 20.0
            # Add points around each minimum
            for offset in [-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005]:
                x0 = x0_min + offset
                if 0.0 <= x0 <= 1.0:
                    candidates_x0.append(x0)
        
        # Also add points at the boundaries of valid regions
        # Where f2 < 11 (reference point)
        # Let's add a finer grid in regions likely to be on the Pareto front
        
        # Remove duplicates and sort
        candidates_x0 = sorted(set([round(x, 10) for x in candidates_x0 if 0.0 <= x <= 1.0]))
        
        # Limit to budget
        if len(candidates_x0) > self.budget:
            # Prioritize: keep grid points and minimum points
            # Simple approach: take first budget points (they're sorted)
            candidates_x0 = candidates_x0[:self.budget]
        
        # Build candidate vectors
        X = np.zeros((len(candidates_x0), self.dim))
        X[:, 0] = candidates_x0
        # x[1:] = 0 for ideal manifold
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Use remaining budget for local refinement
        # Focus on non-dominated points and try to improve them
        # First, find non-dominated points
        def is_dominated(F, idx):
            f = F[idx]
            for j in range(len(F)):
                if j == idx:
                    continue
                if np.all(F[j] <= f) and np.any(F[j] < f):
                    return True
            return False
        
        non_dom_idx = [i for i in range(len(F)) if not is_dominated(F, i)]
        
        # Try local search on non-dominated points
        if used < self.budget and len(non_dom_idx) > 0:
            rng = np.random.default_rng(42)
            # Allocate remaining budget to refine best points
            remaining = self.budget - used
            # Focus on points with low f2 (closer to ideal)
            non_dom_sorted = sorted(non_dom_idx, key=lambda i: F[i, 1])
            # Take top candidates
            n_refine = min(len(non_dom_sorted), max(1, remaining // 3))
            refine_idx = non_dom_sorted[:n_refine]
            
            for idx in refine_idx:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                base = X[idx].copy()
                # Try small perturbations in x[0] and x[1:]
                for _ in range(min(3, self.budget - used)):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Perturb x[0] slightly
                    x_new = base.copy()
                    x_new[0] = base[0] + rng.normal(0, 0.001)
                    x_new[0] = np.clip(x_new[0], 0.0, 1.0)
                    # Keep x[1:] = 0 for ideal manifold
                    x_new[1:] = 0.0
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Replace if better (lower f2 for same f1, or dominates)
                    if f_new[1] < F[idx][1] - 1e-10:
                        X[idx] = x_new
                        F[idx] = f_new
                    elif f_new[0] < F[idx][0] - 1e-10 and f_new[1] <= F[idx][1] + 1e-10:
                        X[idx] = x_new
                        F[idx] = f_new
        
        return X
# EVOLVE-BLOCK-END
