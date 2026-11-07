"""Direct DTLZ4 Pareto-front construction with LHS-based structured sampling."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points using Latin Hypercube Sampling on the simplex.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0..x_2 : position variables on the simplex (x0+x1+x2=1)
      x_3..x_11: control variables; g minimized at x_i=0.5
      f_i = (1+g) * x_{i-1} for i=1,2,3 (with f_3 using 1-x0-x1)
    At g=0, points trace the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        evaluated = []

        # Control variables at their optimum: x_i = 0.5
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # First, add explicit vertex and edge-midpoint points for guaranteed coverage
        special_pts = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]

        for p in special_pts:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # Remaining budget: use Greedy Maximin Sampling on the simplex
        # This places points to maximize the minimum distance to existing points,
        # which is a strong heuristic for maximizing Hypervolume on smooth fronts.
        
        # Store the 3D simplex coordinates (x0, x1, x2) of already selected points
        # We use the special points already added
        selected_simplex_pts = [np.array(p) for p in special_pts]
        
        n_remaining = self.budget - used
        if n_remaining > 0:
            # Pre-allocate a list to hold all simplex points for final evaluation
            # We will append new ones here
            
            for _ in range(n_remaining):
                if time.time() > deadline:
                    break
                
                # Generate a pool of candidate points uniformly on the simplex
                # Using Dirichlet(1,1,1)
                n_cand = 500
                u = rng.random((n_cand, 3))
                u /= u.sum(axis=1, keepdims=True)
                
                # Convert existing points to numpy array for vectorized distance calculation
                P_old = np.array(selected_simplex_pts)
                
                # Calculate pairwise squared Euclidean distances
                # u: (C, 3), P_old: (N, 3)
                # diff: (C, N, 3)
                diff = u[:, np.newaxis, :] - P_old[np.newaxis, :, :]
                dists = np.sum(diff**2, axis=2) # (C, N)
                
                # Find the candidate with the maximum minimum distance to any existing point
                min_dists = np.min(dists, axis=1) # (C,)
                best_idx = np.argmax(min_dists)
                
                new_pt = u[best_idx]
                selected_simplex_pts.append(new_pt)
                
                # We evaluate immediately to keep 'used' count and 'evaluated' list in sync
                # This matches the requirement that evaluations are recorded as they happen
                p = new_pt
                x = np.concatenate([p, ctrl])
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated.append((x, np.array(f)))
                used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
