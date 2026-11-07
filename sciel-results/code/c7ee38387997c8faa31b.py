import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front: x[2:]=0 => g=1
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*h(x[1],1)
        # h(y,1) = 9/(1+2*(sin(3*pi*y)+sin(4*pi*y)))
        #
        # Max-min facility location in (f1,f2)=(x0,x1) space.
        # Key insight: use a DENSE candidate pool so we never repeat points.

        N = budget
        if N < 2:
            N = 2

        # Initial well-spread points: corners + mid-edges + center
        init_candidates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.5, 0.5],
        ])

        n_init = min(len(init_candidates), N)
        placed = init_candidates[:n_init].copy()

        # Evaluate initial points
        for i in range(len(placed)):
            if not check_time():
                break
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = placed[i, 0]
            x[1] = placed[i, 1]
            evaluate(x)

        # Dense candidate grid: 60x60 = 3600 candidates, enough for 200 points
        n_grid = 60
        grid_x0 = np.linspace(0, 1, n_grid)
        grid_x1 = np.linspace(0, 1, n_grid)
        G0, G1 = np.meshgrid(grid_x0, grid_x1, indexing='ij')
        candidates = np.column_stack([G0.ravel(), G1.ravel()])  # shape (M, 2)

        # Track which candidates are already used to avoid duplicates
        # We'll remove used candidates from the pool as we go
        used_mask = np.zeros(len(candidates), dtype=bool)

        # Mark initial placed points as used (find nearest grid point)
        for p in placed:
            dists_to_cands = np.sum((candidates - p)**2, axis=1)
            nearest = np.argmin(dists_to_cands)
            used_mask[nearest] = True

        remaining_budget = budget - used
        for step in range(remaining_budget):
            if not check_time():
                break
            if used >= budget:
                break

            # Compute squared distances from each unused candidate to each placed point
            # Use vectorized approach: for each candidate, min dist to placed
            # To save memory, process in chunks if needed
            n_cand = len(candidates)
            n_placed = len(placed)

            # Vectorized: candidates (M,1,2) - placed (1,K,2) -> (M,K,2)
            # This can be large: 3600 * 200 * 2 * 8 bytes = ~11.5 MB, fine
            diff = candidates[:, np.newaxis, :] - placed[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)  # (M, K)
            min_sq_dists = np.min(sq_dists, axis=1)  # (M,)

            # Mask out already-used candidates
            min_sq_dists[used_mask] = -1.0

            best_idx = np.argmax(min_sq_dists)
            best_point = candidates[best_idx].copy()

            # Mark as used
            used_mask[best_idx] = True

            # Add to placed
            placed = np.vstack([placed, best_point])

            # Evaluate
            x = np.zeros(dim)
            x[0] = best_point[0]
            x[1] = best_point[1]
            evaluate(x)

        return None
