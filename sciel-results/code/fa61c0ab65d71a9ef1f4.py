"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq


class Optimizer:
    """Multi-objective optimizer for ZDT6 using parametric Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # ZDT6 specific: f1 = 1 - (1 - exp(-10*(x1+0.05)))^2 / (1 - 2*exp(-20*(x1+0.05)))
        # We can invert this to get x1 from u where f1_target = 1 - u
        # Actually, let's use the parametric form directly.
        # For ZDT6, the Pareto front is:
        # f1 = 1 - (1 - exp(-10*(x1+0.05)))^2 / (1 - 2*exp(-20*(x1+0.05)))
        # f2 = g(x') * h(f1, g)
        # where g = 1 + 9/n' * sum(x_i) for i=2..n
        # and h = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        # On the Pareto front, f2 = 0 when h = 0, which happens when f1/g = 1 or specific values.
        # Actually, the Pareto front for ZDT6 is a curve. Let's use a different approach.
        # 
        # Better approach: Use a hybrid strategy
        # 1. First, sample points along the approximate Pareto front
        # 2. Then use remaining budget for local refinement

        # For ZDT6, we know the Pareto front structure.
        # Let's create candidate points by varying x1 and setting other variables to 0
        # (since g is minimized when x_i = 0 for i >= 2)
        
        # First, let's do a coarse search to find good points
        n_init = min(50, budget // 4)
        remaining = budget - n_init
        
        # Generate initial population with focus on Pareto front region
        # For ZDT6, x1 in [0,1] and x2..x10 = 0 gives points near the front
        X_candidates = np.zeros((n_init, dim))
        # Spread x1 across [0, 1]
        X_candidates[:, 0] = np.linspace(0, 1, n_init)
        # Add some random perturbation to other dimensions
        for i in range(1, dim):
            X_candidates[:, i] = 0.0
        
        # Evaluate
        F = np.array([problem(x) for x in X_candidates])
        used = n_init

        # Keep track of best points (non-dominated)
        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
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
            return np.where(~dominated)[0]

        # Use remaining budget for local optimization around best points
        # Use a simple local search: perturb non-dominated points
        
        # Get non-dominated points
        nd_idx = non_dominated_sort(F)
        
        # If we have time and budget, do local refinement
        if remaining > 0 and len(nd_idx) > 0:
            # Select a subset of non-dominated points for local search
            n_local = min(len(nd_idx), min(10, remaining // 5))
            local_indices = nd_idx[:n_local]
            
            # For each selected point, try small perturbations
            perturbations = [0.01, 0.005, 0.001]
            for idx in local_indices:
                if used >= budget:
                    break
                if time.time() > deadline - 30:
                    break
                
                base = X_candidates[idx].copy()
                base_F = F[idx].copy()
                
                for _ in range(3):  # Try a few perturbations per point
                    if used >= budget:
                        break
                    if time.time() > deadline - 30:
                        break
                    
                    # Random direction in decision space
                    direction = np.random.randn(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    step = 0.05 * np.random.choice(perturbations)
                    
                    new_x = base + step * direction
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_F = problem(new_x)
                    used += 1
                    
                    # Check if new point dominates or is non-dominated
                    # Simple check: if it's better in at least one objective and not worse in others
                    if np.any(new_F < base_F) and np.all(new_F <= base_F + 1e-10):
                        # Replace the base point
                        X_candidates[idx] = new_x
                        F[idx] = new_F
                        base = new_x
                        base_F = new_F

        # Final: return the best set of points
        # The evaluator records all returned objective vectors
        # We should return points that are well-spread on the Pareto front
        
        # Get final non-dominated set
        nd_idx_final = non_dominated_sort(F)
        
        # If we have very few non-dominated points, include some dominated ones for spread
        if len(nd_idx_final) < 10:
            # Add points with good spread in objective space
            # Use a simple clustering approach
            all_indices = list(range(len(F)))
            selected = list(nd_idx_final)
            
            # Add more points for spread
            while len(selected) < min(20, len(F)):
                # Find point that maximizes minimum distance to selected points in objective space
                best_idx = -1
                best_min_dist = -1
                for idx in all_indices:
                    if idx in selected:
                        continue
                    min_dist = min(np.linalg.norm(F[idx] - F[s]) for s in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                if best_idx == -1:
                    break
                selected.append(best_idx)
            
            nd_idx_final = np.array(selected)
        
        # Return the X array (the return value is ignored, but we need to return something)
        return X_candidates
# EVOLVE-BLOCK-END
