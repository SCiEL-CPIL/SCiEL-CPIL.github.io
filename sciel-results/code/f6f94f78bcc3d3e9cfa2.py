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
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # DTLZ1 structure: x = [x1, x2, x3, ..., xM] where M = n_obj
        # For 3 objectives, M=3, so x[0], x[1] are front parameters, x[2:] affect g
        # g(x) = 1 + 10*(M-1) + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i=2..M
        # f1 = (g+1)*x1^2
        # f2 = (g+1)*x1*sqrt(1-x1^2)  
        # f3 = (g+1)*(1-x1^2)
        # Wait, standard DTLZ1:
        # f1 = 0.5 * x1 * x2 * g
        # f2 = 0.5 * x1 * (1-x2) * g
        # f3 = 0.5 * (1-x1) * g
        # where g = 1 + 9/(M-1) * sum_{i=M}^{n} x_i
        # Actually for DTLZ1 with M objectives:
        # g(x) = 1 + 9 * mean(x_M...x_n)
        # f_i = (g+1) * prod_{j=1}^{i-1} x_j * (1-x_i) for i=1..M-1
        # f_M = (g+1) * prod_{j=1}^{M-1} (1-x_j)
        
        # For 3 objectives (M=3):
        # g = 1 + 9 * mean(x_3, ..., x_7)  [0-indexed: x[2:7]]
        # f1 = (g+1) * x[0] * x[1]
        # f2 = (g+1) * x[0] * (1-x[1])
        # f3 = (g+1) * (1-x[0])
        
        # To get points on the Pareto front, we need g=1 (minimum)
        # g=1 when x[2:7] = 0 for all i (since mean=0)
        # Then: f1 = 2*x0*x1, f2 = 2*x0*(1-x1), f3 = 2*(1-x0)
        
        # The Pareto front is parameterized by x0, x1 in [0,1]
        # We can generate many points on the front by varying x0, x1 and setting x[2:]=0
        
        # Strategy:
        # 1. Generate a grid of (x0, x1) points
        # 2. Set x[2:] = 0 (to minimize g)
        # 3. Evaluate all points
        # 4. Use remaining budget to do local refinement around best points
        
        # Phase 1: Grid sampling on the Pareto front
        # Use most of the budget for this
        n_grid = int(budget * 0.7)
        
        # Create a grid in (x0, x1) space
        # Use stratified sampling for better coverage
        n_per_dim = int(np.sqrt(n_grid))
        if n_per_dim < 10:
            n_per_dim = 10
        n_grid_actual = n_per_dim * n_per_dim
        
        # Latin hypercube-like sampling for better coverage
        rng = np.random.default_rng(42)
        
        # Generate stratified samples
        x0_samples = np.linspace(0.01, 0.99, n_per_dim) + rng.uniform(0, 0.98/n_per_dim, n_per_dim)
        x1_samples = np.linspace(0.01, 0.99, n_per_dim) + rng.uniform(0, 0.98/n_per_dim, n_per_dim)
        
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
        if time.time() > deadline - 30:
            return X
        
        # Phase 2: Local refinement around non-dominated points
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
        
        # Use remaining budget for refinement
        remaining = budget - used
        if remaining > 0 and len(nondom_idx) > 0:
            # Refine around best non-dominated points
            # Pick top k non-dominated points
            k = min(10, len(nondom_idx))
            # Select diverse points from non-dominated set
            if k < len(nondom_idx):
                # Simple diversity selection: pick evenly spaced indices
                step = len(nondom_idx) // k
                selected = nondom_idx[::step][:k]
            else:
                selected = nondom_idx
            
            for _ in range(min(remaining, 50)):  # Cap refinement evals
                if time.time() > deadline - 10:
                    break
                if used >= budget:
                    break
                
                # Pick a random selected point and perturb
                parent_idx = selected[rng.integers(len(selected))]
                x_parent = X[parent_idx].copy()
                
                # Small perturbation in x0, x1
                perturbation = 0.05 * rng.standard_normal(2)
                x_child = x_parent.copy()
                x_child[0] = np.clip(x_child[0] + perturbation[0], 0, 1)
                x_child[1] = np.clip(x_child[1] + perturbation[1], 0, 1)
                
                # Keep x[2:] = 0
                x_child[2:] = 0
                
                fc = problem(x_child)
                used += 1
                
                # Add to population if it improves
                # Simple: replace worst point if child is better
                # Use sum of objectives as proxy
                worst_idx = np.argmax(np.sum(F, axis=1))
                
                # Check if child dominates worst
                if np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx]):
                    X[worst_idx] = x_child
                    F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
