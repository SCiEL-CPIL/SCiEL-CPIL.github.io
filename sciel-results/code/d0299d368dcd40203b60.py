"""Model-and-optimize for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Space-filling and distance-based optimization:
    1) Generate a large Halton sequence.
    2) Iteratively select points that maximize the minimum distance to existing points in objective space.
    3) Local search to refine the selected points.
    4) Interpolation and remaining budget for random restarts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- Halton Sequence Generator ---
        def halton(dim, num):
            seq = np.zeros((num, dim))
            for i in range(num):
                for d in range(dim):
                    f = 1.0
                    r = 0.0
                    base = d + 2
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    seq[i, d] = r
            return seq

        # --- 1) Generate Candidate Pool ---
        # Generate a large pool of space-filling points
        n_candidates = 2000
        halton_pts = halton(dim, n_candidates)
        # Mix in some random points for robustness
        rand_pts = rng.uniform(lo, hi, size=(200, dim))
        candidates = np.vstack([halton_pts, rand_pts])
        
        # Add specific corners and center to ensure bounds are explored
        specials = [lo, hi, (lo+hi)/2.0,
                    np.array([0,0,1,1]), np.array([0,1,0,1]),
                    np.array([0,1,1,0]), np.array([1,0,0,1]),
                    np.array([1,0,1,0]), np.array([1,1,0,0])]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)
        candidates = np.vstack([specials, candidates])

        # --- 2) Greedy Selection based on Objective Space Distance ---
        selected_indices = []
        selected_X = []
        selected_Y = []
        
        # Start with a few random points from the pool to initialize
        init_indices = rng.choice(len(candidates), size=min(10, len(candidates)), replace=False)
        
        for idx in init_indices:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            x = candidates[idx]
            f = eval_real(x)
            selected_indices.append(idx)
            selected_X.append(x)
            selected_Y.append(f)

        # Iteratively select the point that maximizes the minimum distance to existing selected points in objective space
        # We use the candidate pool to find the next best point.
        # To make this efficient, we pre-compute distances from all candidates to the current selected set.
        
        # We will do this for a fixed number of steps or until budget is low
        max_selection_steps = 50
        
        for step in range(max_selection_steps):
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            
            # If we have no selected points, pick a random one
            if len(selected_Y) == 0:
                idx = rng.integers(0, len(candidates))
                x = candidates[idx]
                f = eval_real(x)
                selected_indices.append(idx)
                selected_X.append(x)
                selected_Y.append(f)
                continue

            # Calculate min distance from each candidate to the current selected set (in objective space)
            # We only care about candidates not already selected
            remaining_mask = np.ones(len(candidates), dtype=bool)
            for idx in selected_indices:
                remaining_mask[idx] = False
            
            remaining_candidates = candidates[remaining_mask]
            if len(remaining_candidates) == 0:
                break
                
            selected_Y_arr = np.vstack(selected_Y)
            
            # Compute distances: (n_candidates, n_obj) vs (n_selected, n_obj)
            # dist(i, j) = ||f_i - f_j||
            # We want for each candidate i: min_j dist(i, j)
            # Then pick i with max min_dist
            
            # Efficient calculation:
            # dists = sqrt(sum((candidates[:, None, :] - selected_Y_arr[None, :, :])**2, axis=2))
            # min_dists = np.min(dists, axis=1)
            
            # To save memory/time, we can do it in chunks if needed, but 2000x50 is small.
            diff = remaining_candidates[:, None, :] - selected_Y_arr[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            # Find the candidate with the maximum minimum distance
            best_local_idx = np.argmax(min_dists)
            best_global_idx = np.where(remaining_mask)[0][best_local_idx]
            
            x = candidates[best_global_idx]
            f = eval_real(x)
            selected_indices.append(best_global_idx)
            selected_X.append(x)
            selected_Y.append(f)

        # --- 3) Local Real Optimization ---
        def local_hill(x, steps=20):
            best = np.asarray(x, dtype=float).copy()
            bestf = eval_real(best)
            for i in range(steps):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                radius = 0.05 + 0.10 * (1.0 - i / max(steps, 1))
                step = rng.normal(0.0, radius, size=dim)
                cand = np.clip(best + step, lo, hi)
                f = eval_real(cand)
                if np.all(f <= bestf) and np.any(f < bestf):
                    best = cand
                    bestf = f
            return best, bestf

        # Refine the selected points
        for x in selected_X:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            b, bf = local_hill(x, steps=15)
            # We don't strictly need to track pareto set for the return, but it helps for interpolation
            # The evaluator records all returned vectors, so we just need to call problem via eval_real
            # Wait, the task says "Every returned objective vector is recorded".
            # My eval_real calls problem(x) and returns f. The evaluator records the return of __call__?
            # "The return value of __call__ is ignored."
            # "Every returned objective vector is recorded" -> This implies the evaluator hooks into the problem function or the Optimizer returns them?
            # Re-reading: "problem(x) ... returns 3 objective values."
            # "Every returned objective vector is recorded" -> The evaluator likely wraps `problem` or `evaluator` to record calls.
            # Actually, the prompt says: "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
            # "Every returned objective vector is recorded" -> This is slightly ambiguous. Does it mean the return of `problem`?
            # "problem(x) ... returns 3 objective values."
            # Usually in these tasks, the `problem` function is the one being called, and the evaluator records the outputs of `problem`.
            # So I just need to ensure I call `problem` (via `eval_real`) within budget.
            pass

        # --- 4) Interpolation ---
        # Use the selected points to interpolate
        if len(selected_X) >= 2:
            used = set()
            for _ in range(30):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                i = rng.integers(0, len(selected_X))
                j = rng.integers(0, len(selected_X))
                if i == j:
                    continue
                xi = selected_X[i]
                xj = selected_X[j]
                t = rng.uniform(0.15, 0.85)
                cand = np.clip((1.0-t)*xi + t*xj, lo, hi)
                key = tuple(np.round(cand, 6))
                if key in used:
                    continue
                used.add(key)
                f = eval_real(cand)

        # --- 5) Remaining Budget ---
        while self.eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            try:
                b, bf = local_hill(x, steps=10)
            except Exception:
                break

        return None
# EVOLVE-BLOCK-END
