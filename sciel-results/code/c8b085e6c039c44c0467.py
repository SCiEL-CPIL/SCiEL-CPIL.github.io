# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer exploiting WFG9 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # Strategy: Exploit WFG9 structure
        # First 2 variables define the Pareto front shape
        # Remaining 10 variables affect distance from front
        # Use dense grid on first 2, optimize remaining locally
        
        midpoints = (lo + hi) / 2.0
        
        # Grid on first 2 variables: 12x12 = 144 points
        n1, n2 = 12, 12
        grid_size = n1 * n2
        
        if grid_size > budget:
            scale = np.sqrt(budget / grid_size)
            n1 = max(2, int(n1 * scale))
            n2 = max(2, int(n2 * scale))
            grid_size = n1 * n2

        x0_vals = np.linspace(lo[0], hi[0], n1)
        x1_vals = np.linspace(lo[1], hi[1], n2)
        
        points = []
        F = []
        eval_count = 0
        
        # Phase 1: Grid search on first 2 variables with midpoints for rest
        for x0 in x0_vals:
            if time.time() > deadline - 30:
                break
            for x1 in x1_vals:
                if eval_count >= budget or time.time() > deadline - 30:
                    break
                x = np.copy(midpoints)
                x[0] = x0
                x[1] = x1
                f = problem(x)
                points.append(x)
                F.append(f)
                eval_count += 1
        
        # Find non-dominated points
        F = np.array(F)
        nd_indices = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)
        
        # Phase 2: Local refinement of non-dominated points
        remaining = budget - eval_count
        if remaining > 0 and nd_indices:
            # Distribute remaining budget among non-dominated points
            n_refine = min(len(nd_indices), remaining)
            points_to_refine = nd_indices[:n_refine]
            evals_per_point = remaining // max(1, n_refine)
            
            for idx in points_to_refine:
                if eval_count >= budget or time.time() > deadline - 10:
                    break
                base = points[idx].copy()
                for _ in range(evals_per_point):
                    if eval_count >= budget or time.time() > deadline - 10:
                        break
                    # Random perturbation in all dimensions
                    pert = np.random.uniform(-0.15, 0.15, size=dim) * (hi - lo)
                    new_x = np.clip(base + pert, lo, hi)
                    f_new = problem(new_x)
                    points.append(new_x)
                    F = np.vstack([F, [f_new]])
                    eval_count += 1
        
        return points
# EVOLVE-BLOCK-END
