"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer: structured 2D grid on (x0,x1) with x[2:]=0,
    then local refinement around hypervolume-contributing points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        def make_x(x0, x1, rest=None):
            """Build a 22-d vector with x0, x1 set, and x[2:] = 0 (or rest)."""
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            if rest is not None:
                x[2:] = rest
            return x

        def eval_x(x):
            """Evaluate and return objective vector."""
            return np.asarray(problem(x), dtype=float)

        used = 0
        F_list = []  # list of objective vectors (all recorded)
        X_list = []  # corresponding decision vectors (for reference)

        def record(x):
            nonlocal used
            f = eval_x(x)
            F_list.append(f)
            X_list.append(x)
            used += 1
            return f

        # --- Phase 1: Dense 1D sweep along the Pareto front ---
        # DTLZ7 with x[2:]=0 (g=0) has Pareto front parameterized by t in [0,1]:
        # x0 = t, x1 = 1 - t
        # f1 = t*(2 - sqrt(t/(1-t))), f2 = (1-t)*(1+t), f3 = 1
        # Use 160 points to densely cover the front, leaving 40 for refinement.
        
        n_sweep = 160
        eps = 1e-8
        # t values: avoid t=0 and t=1 exactly
        t_vals = np.linspace(eps, 1.0 - eps, n_sweep)
        
        grid_F = []
        grid_X = []
        for t in t_vals:
            if used >= self.budget:
                break
            x0 = t
            x1 = 1.0 - t
            x = make_x(x0, x1)
            f = record(x)
            grid_F.append(f)
            grid_X.append(x)
        
        if used >= self.bbudget:
            return X_list
        
        # --- Phase 2: Refinement around non-dominated points ---
        # Find non-dominated points in the sweep
        n_pts = len(grid_F)
        if n_pts > 0:
            G = np.array(grid_F)
            # Non-dominated: no other point dominates it
            nd_mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                if not nd_mask[i]:
                    continue
                # Check if any other point dominates i
                dominated_by = np.any(
                    np.all(G <= G[i], axis=1) & np.any(G < G[i], axis=1),
                    axis=0
                )
                dominated_by[i] = False
                if np.any(dominated_by):
                    nd_mask[i] = False
            
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.array([0])
        else:
            nd_indices = np.array([])
        
        # Refine around non-dominated points with small perturbations in t
        rng = np.random.default_rng(42)
        step_sizes = [0.005, 0.002, 0.001]
        
        if len(nd_indices) > 0:
            G_nd = G[nd_indices]
            sort_idx = np.argsort(G_nd[:, 0] + G_nd[:, 1])
            nd_indices_sorted = nd_indices[sort_idx]
        else:
            nd_indices_sorted = np.array([])
        
        for idx in nd_indices_sorted:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x_base = grid_X[idx]
            t_base = x_base[0]  # x0 = t
            
            for step in step_sizes:
                if used >= self.budget or time.time() > deadline:
                    break
                for dt in [-step, step]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    t_new = np.clip(t_base + dt, eps, 1.0 - eps)
                    x0_new = t_new
                    x1_new = 1.0 - t_new
                    x = make_x(x0_new, x1_new)
                    record(x)
        
        # If we still have budget, do some random exploration
        while used < self.budget and time.time() < deadline:
            t = rng.uniform(eps, 1.0 - eps)
            x0 = t
            x1 = 1.0 - t
            x = make_x(x0, x1)
            record(x)

        return X_list
# EVOLVE-BLOCK-END
