"""Global basin search + Rastrigin-aware local refinement for BBOB f4 (Büche-Rastrigin)."""
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
        deadline = start_time + 280  # safety margin before 300s limit

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
        def local_search(x0, f0, max_calls=60, fine=False):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            else:
                step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05]

            for step in step_sizes:
                if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                    break
                improved = True
                while improved and calls_used < max_calls and call_count < budget - 5 and time_ok():
                    improved = False
                    for i in range(dim):
                        if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                            break
                        base = x[i]
                        for delta in [step, -step]:
                            if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                                break
                            c = base + delta
                            if not (-4.95 <= c <= 4.95):
                                continue
                            x[i] = c
                            cf = do_call(x)
                            calls_used += 1
                            if cf < f:
                                f = cf
                                improved = True
                            else:
                                x[i] = base
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ---------- Phase 0: Initial exploration ----------
        init_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-2.0, -1.0, 1.0, 2.0):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        while len(init_pts) < 20:
            init_pts.append(rng.uniform(-4.0, 4.0, dim))

        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)

        # ---------- Phase 1: Basin refinement with jumps ----------
        reserve = 40
        max_basins = 5
        for _ in range(max_basins):
            if call_count >= budget - reserve or not time_ok():
                break

            # Refine current best
            local_search(best_x, best_f, max_calls=50, fine=False)

            if call_count >= budget - reserve or not time_ok():
                break

            # Try to jump to a new basin
            n_jump = rng.integers(1, dim + 1)
            jump_coords = rng.choice(dim, size=n_jump, replace=False)
            jump = np.zeros(dim)
            for j in jump_coords:
                jump[j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.5)
            seed = np.clip(best_x + jump, -4.95, 4.95)
            f_seed = do_call(seed)

            # Quick refine from new seed
            if call_count < budget - reserve and time_ok():
                local_search(seed, f_seed, max_calls=25, fine=False)

        # ---------- Phase 2: Final fine-tune ----------
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=200, fine=True)

        return best_x, best_f
