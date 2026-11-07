"""Structure-exploiting grid-based basin identification + Rastrigin-aware local refinement for BBOB f4."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # ---------- Rastrigin-aware local search ----------
        def local_search(x0, f0, max_calls=120, fine=False):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                offsets = [0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]
            else:
                offsets = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25, 0.1, -0.1]

            for _ in range(max_calls):
                if calls_used >= max_calls or call_count >= budget - 2 or not time_ok():
                    break
                improved = False
                for i in range(dim):
                    if calls_used >= max_calls or call_count >= budget - 2 or not time_ok():
                        break
                    base = x[i]
                    local_best = None
                    local_best_f = f
                    for off in offsets:
                        if calls_used >= max_calls or call_count >= budget - 2 or not time_ok():
                            break
                        c = base + off
                        if not (-4.95 <= c <= 4.95):
                            continue
                        x[i] = c
                        cf = do_call(x)
                        calls_used += 1
                        if cf < local_best_f:
                            local_best_f = cf
                            local_best = c
                    if local_best is not None:
                        x[i] = local_best
                        f = local_best_f
                        improved = True
                    else:
                        x[i] = base
                if not improved:
                    break

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ---------- Phase 0: Coarse grid sampling to map ridge structure ----------
        # Büche-Rastrigin has periodic ridges with period 2. Sample a coarse grid
        # to identify which regions contain low function values (promising basins).
        # Grid spacing of 2.0 aligns with the ridge period.
        grid_spacing = 2.0
        grid_points = []
        for i in range(-1, 3):  # -1, 0, 1, 2
            for j in range(-1, 3):
                for k in range(-1, 3):
                    for l in range(-1, 3):
                        for m in range(-1, 3):
                            p = np.array([i, j, k, l, m], dtype=float) * grid_spacing
                            if np.all(p >= -4.95) and np.all(p <= 4.95):
                                grid_points.append(p)
        
        # Also add midpoints between grid points to catch basins between ridges
        for i in range(-1, 3):
            for j in range(-1, 3):
                for k in range(-1, 3):
                    for l in range(-1, 3):
                        for m in range(-1, 3):
                            p = (np.array([i, j, k, l, m], dtype=float) + 0.5) * grid_spacing
                            if np.all(p >= -4.95) and np.all(p <= 4.95):
                                grid_points.append(p)

        # Shuffle grid points to avoid bias toward corner cells
        rng.shuffle(grid_points)

        # Evaluate a subset of grid points (limited by budget)
        grid_eval_budget = min(len(grid_points), 80)
        grid_results = []
        for p in grid_points[:grid_eval_budget]:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)
            grid_results.append((f, p.copy()))

        # Sort by function value and take the top candidates
        grid_results.sort(key=lambda t: t[0])
        top_grid = [p for _, p in grid_results[:15]]

        # ---------- Phase 1: Generate candidate minima using periodicity ----------
        # For each promising grid point, generate candidates by shifting by
        # integer multiples of the ridge period (2.0) in each coordinate,
        # clipped to the valid domain.
        candidates = set()
        for gp in top_grid:
            # The grid point itself
            candidates.add(tuple(np.round(gp, 6)))
            # Shift by ±1.0 (half period) in each coordinate
            for i in range(dim):
                for shift in [-1.0, 1.0]:
                    c = gp.copy()
                    c[i] += shift
                    if np.all(c >= -4.95) and np.all(c <= 4.95):
                        candidates.add(tuple(np.round(c, 6)))
            # Shift by ±0.5 (quarter period) in each coordinate
            for i in range(dim):
                for shift in [-0.5, 0.5]:
                    c = gp.copy()
                    c[i] += shift
                    if np.all(c >= -4.95) and np.all(c <= 4.95):
                        candidates.add(tuple(np.round(c, 6)))
        
        # Also add some random points for robustness
        for _ in range(10):
            p = rng.uniform(-4.5, 4.5, dim)
            candidates.add(tuple(np.round(p, 6)))

        candidates = list(candidates)
        rng.shuffle(candidates)

        # Evaluate and refine the most promising candidates
        reserve = 30
        for cand in candidates:
            if call_count >= budget - reserve or not time_ok():
                break
            x0 = np.array(cand, dtype=float)
            x0 = np.clip(x0, -4.95, 4.95)
            f0 = do_call(x0)
            
            # Quick local refinement
            if call_count < budget - reserve and time_ok():
                x, f = local_search(x0, f0, max_calls=40, fine=False)
        
        # ---------- Phase 2: Multi-basin refinement ----------
        max_basins = 6
        for b in range(max_basins):
            if call_count >= budget - reserve or not time_ok():
                break

            # Refine current best basin
            x, f = local_search(best_x, best_f, max_calls=80, fine=False)

            if call_count >= budget - reserve or not time_ok():
                break

            # Jump to a new region
            n_jump = rng.integers(1, dim + 1)
            jump_coords = rng.choice(dim, size=n_jump, replace=False)
            jump = np.zeros(dim)
            for j in jump_coords:
                jump[j] = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
            seed = best_x + jump
            seed = np.clip(seed, -4.95, 4.95)
            f_seed = do_call(seed)

            if call_count < budget - reserve and time_ok():
                x2, f2 = local_search(seed, f_seed, max_calls=50, fine=False)

        # ---------- Phase 3: Final fine-tune ----------
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=200, fine=True)

        return best_x, best_f
