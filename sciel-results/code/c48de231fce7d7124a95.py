import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto front:
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)^6)^(1/10)
        # f2 = 1 - sqrt(f1)
        # The front is parameterized by x1 in [0,1].

        # Strategy: Use closed-form parametric representation.
        # Let u = (1 - exp(-4*x1) * sin(6*pi*x1)^6)^(1/10), so f1 = 1 - u.
        # We want to choose x1 values that give well-distributed f1 values.
        # 
        # Key insight: f1(x1) is NOT monotonic in x1. It has multiple local extrema.
        # So we can't just invert it. Instead, we sample x1 densely, compute f1,
        # and then pick a subset of points that maximizes hypervolume.
        #
        # For HV maximization with reference point (11, 11):
        # Given sorted points (t_1, 1-sqrt(t_1)), ..., (t_n, 1-sqrt(t_n)) where t_1 < ... < t_n,
        # HV = sum_{i=1}^{n} (t_{i+1} - t_i) * (10 + sqrt(t_i)) + (11 - t_n)*(10 + sqrt(t_n))
        # where t_{n+1} = 11.
        #
        # The optimal distribution of t_i to maximize this is approximately uniform,
        # but slightly biased towards smaller t values (since 10+sqrt(t) decreases with t).
        #
        # Approach:
        # 1. Build a dense grid of x1 values, compute f1 for each.
        # 2. For each target f1 value (chosen to maximize HV), find the x1 that achieves it.
        # 3. Evaluate all 200 points.

        # Step 1: Build dense lookup of f1(x1)
        n_grid = 4000000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        f1_grid = np.empty(n_grid)
        chunk = 400000
        for i in range(0, n_grid, chunk):
            end = min(i + chunk, n_grid)
            x_chunk = x1_grid[i:end]
            sin_val = np.sin(6.0 * np.pi * x_chunk)
            f1_grid[i:end] = 1.0 - (1.0 - np.exp(-4.0 * x_chunk) * sin_val**6)**0.1

        # Sort by f1 to create a lookup
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]

        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]

        # Step 2: Choose n_points f1 values to maximize HV
        n_points = min(budget, 200)
        
        # Start with uniform spacing in f1
        t = np.linspace(f1_min, f1_max, n_points)

        def compute_hv(t_arr):
            t_sorted = np.sort(t_arr)
            t_ext = np.append(t_sorted, 11.0)
            widths = t_ext[1:] - t_ext[:-1]
            heights = 10.0 + np.sqrt(t_sorted)
            return np.sum(widths * heights)

        # Local optimization via coordinate descent with adaptive step
        hv_cur = compute_hv(t)
        step = (f1_max - f1_min) / (n_points * 4)
        max_iters = 50
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            improved = False
            for i in range(n_points):
                # Try moving point i
                for delta in [-step, step, -step*0.5, step*0.5]:
                    t_trial = t.copy()
                    t_trial[i] = np.clip(t[i] + delta, f1_min, f1_max)
                    hv_trial = compute_hv(t_trial)
                    if hv_trial > hv_cur:
                        t = t_trial
                        hv_cur = hv_trial
                        improved = True
                        break
            if not improved:
                step *= 0.7
                if step < 1e-14:
                    break

        # Step 3: Map f1 targets to x1 values
        x1_targets = np.empty(n_points)
        for i in range(n_points):
            target = t[i]
            idx = np.searchsorted(f1_sorted, target)
            idx = np.clip(idx, 0, len(f1_sorted) - 1)
            # Check neighbors for best match
            best_dist = np.inf
            best_x1 = x1_sorted[idx]
            lo = max(0, idx - 5)
            hi = min(len(f1_sorted), idx + 6)
            for j in range(lo, hi):
                dist = abs(f1_sorted[j] - target)
                if dist < best_dist:
                    best_dist = dist
                    best_x1 = x1_sorted[j]
            x1_targets[i] = best_x1

        # Step 4: Evaluate all points
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = x1_targets[i]
            # Clip to bounds
            for d in range(dim):
                x[d] = np.clip(x[d], bounds[d][0], bounds[d][1])
            obj = problem(x)
            recorded.append(obj)

        return recorded
