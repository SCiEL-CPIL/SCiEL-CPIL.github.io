# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Tchebycheff scalarizations and simulated annealing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)

        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 15

        # Phase 1: Generate Tchebycheff weight vectors
        # For 3 objectives, generate weights on the simplex
        n_weights = 50
        weights = []
        # Generate points on the 2D simplex (w0 + w1 + w2 = 1, all >= 0)
        n_per_edge = 5  # 5 points per edge -> C(5+2-1, 2-1) = 15 unique, plus interior
        # Use a more systematic approach: generate on a grid and filter
        n_grid = 6
        for i in range(n_grid):
            for j in range(n_grid):
                w0 = i / (n_grid - 1)
                w1 = j / (n_grid - 1)
                w2 = 1.0 - w0 - w1
                if w2 >= -1e-10:
                    w = np.array([w0, w1, max(0, w2)])
                    w = w / (np.sum(w) + 1e-15)
                    weights.append(w)
        
        # Add some random weights for diversity
        for _ in range(20):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        
        weights = np.array(weights)
        
        # Select the first n_weights weights
        if len(weights) > n_weights:
            # Uniformly sample from the generated weights
            idx = np.linspace(0, len(weights) - 1, n_weights, dtype=int)
            weights = weights[idx]
        
        # Ensure we have exactly n_weights
        while len(weights) < n_weights:
            w = rng.dirichlet(np.ones(n_obj))
            weights = np.vstack([weights, w.reshape(1, -1)])
        weights = weights[:n_weights]
        
        # Phase 2: Solve each Tchebycheff subproblem using simulated annealing
        X = np.zeros((n_weights, dim))
        F = np.zeros((n_weights, n_obj))
        used = 0
        
        # Estimate objective ranges for normalization
        # Do a quick random sample to estimate ranges
        n_est = 10
        X_est = lo + rng.random((n_est, dim)) * (hi - lo)
        F_est = np.array([problem(x) for x in X_est])
        used += n_est
        
        f_min = F_est.min(axis=0)
        f_max = F_est.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0
        
        # For each weight vector, run simulated annealing
        sa_budget_per_weight = (budget - used) // n_weights
        if sa_budget_per_weight < 5:
            sa_budget_per_weight = 5
            n_weights = max(1, (budget - used) // 5)
            weights = weights[:n_weights]
        
        for w_idx in range(n_weights):
            if used >= budget or is_deadline():
                break
            
            w = weights[w_idx]
            
            # Initialize with a random point
            x = lo + rng.random(dim) * (hi - lo)
            f = problem(x)
            used += 1
            
            # Simulated annealing
            T_init = 1.0
            T_min = 1e-6
            cooling_rate = (T_min / T_init) ** (1.0 / max(1, sa_budget_per_weight - 1))
            T = T_init
            
            for sa_iter in range(sa_budget_per_weight - 1):
                if used >= budget or is_deadline():
                    break
                
                # Generate neighbor
                step_size = 0.1 * (hi - lo)
                x_new = x + step_size * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Compute Tchebycheff objective
                def tcheb(f_val, w_vec):
                    return np.max(w_vec * np.abs(f_val - f_min) / f_range)
                
                obj_curr = tcheb(f, w)
                obj_new = tcheb(f_new, w)
                delta = obj_new - obj_curr
                
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-15)):
                    x = x_new
                    f = f_new
                
                T *= cooling_rate
            
            X[w_idx] = x
            F[w_idx] = f
        
        # Phase 3: Refine with local search on the Pareto front
        # Find non-dominated points
        nd_mask = self._non_dominated_mask(F)
        nd_idx = np.where(nd_mask)[0]
        
        # Use remaining budget for local refinement
        remaining = budget - used
        if remaining > 0 and len(nd_idx) > 0:
            # Select the most spread out non-dominated points for refinement
            # Use a simple strategy: refine each non-dominated point
            n_refine = min(len(nd_idx), max(1, remaining // 10))
            if n_refine > 0:
                # Uniformly sample from non-dominated indices
                refine_idx = np.linspace(0, len(nd_idx) - 1, n_refine, dtype=int)
                refine_idx = nd_idx[refine_idx]
                
                budget_per_refine = remaining // n_refine
                
                for r_idx in refine_idx:
                    if used >= budget or is_deadline():
                        break
                    
                    x = X[r_idx].copy()
                    f = F[r_idx].copy()
                    
                    # Local search with decreasing step size
                    for ls_iter in range(budget_per_refine):
                        if used >= budget or is_deadline():
                            break
                        
                        # Try to improve by moving toward the reference point in objective space
                        # Use a simple perturbation
                        step_size = 0.05 * (hi - lo) * (1.0 - ls_iter / max(1, budget_per_refine))
                        x_new = x + step_size * rng.standard_normal(dim)
                        x_new = np.clip(x_new, lo, hi)
                        
                        f_new = problem(x_new)
                        used += 1
                        
                        # Accept if it's not dominated by the current point
                        if not (np.all(f_new >= f) and np.any(f_new > f)):
                            x = x_new
                            f = f_new
                    
                    X[r_idx] = x
                    F[r_idx] = f
        
        return X
# EVOLVE-BLOCK-END
