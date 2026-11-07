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
        deadline = start_time + 285  # safety margin before 300s limit

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
        # Büche-Rastrigin = sum_i [10*(x_i^2 - 1) + cos(2*pi*x_i)] + 10*dim.
        # The cosine term creates ridges; a coordinate move that can jump across
        # a ridge (not just step) is the key to escaping local minima.
        def local_search(x0, f0, max_calls=150, fine=False):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                offsets = [0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]
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
            return x, f, calls_used

        # ---------- Phase 0: Initial exploration ----------
        # Stratified grid + random to cover the domain broadly
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

        # ---------- Phase 1: Multi-basin refinement with structured jumps ----------
        # Iterate: refine best basin, then jump to a new region using
        # structured perturbations (Rastrigin-aware) or random restarts.
        reserve = 40
        max_basins = 8

        for basin_iter in range(max_basins):
            if call_count >= budget - reserve or not time_ok():
                break

            # Refine the current best basin
            x, f, iters = local_search(best_x, best_f, max_calls=120, fine=False)

            if call_count >= budget - reserve or not time_ok():
                break

            # Generate a jump to explore a new basin
            # Use a mix of structured (Rastrigin-aware) and random jumps
            jump_type = rng.random()
            if jump_type < 0.4:
                # Structured jump: perturb 1-2 coordinates by ridge-width multiples
                n_jump = rng.integers(1, min(3, dim) + 1)
                jump_coords = rng.choice(dim, size=n_jump, replace=False)
                seed = best_x.copy()
                for j in jump_coords:
                    # Jump by ~1.0 (one cosine period) in a random direction
                    seed[j] += rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.5)
            elif jump_type < 0.7:
                # Moderate random perturbation
                seed = best_x + rng.normal(0, 0.8, dim)
            else:
                # Full random restart
                seed = rng.uniform(-4.0, 4.0, dim)

            seed = np.clip(seed, -4.95, 4.95)
            f_seed = do_call(seed)

            # Quick local refine from the new seed
            if call_count < budget - reserve and time_ok():
                x2, f2, _ = local_search(seed, f_seed, max_calls=80, fine=False)

        # ---------- Phase 2: Final high-precision polish ----------
        # Use a tight local search with very small steps to squeeze out
        # the last bit of improvement near the best point found.
        if call_count < budget - 1 and time_ok():
            # First, a medium-fine pass
            local_search(best_x, best_f, max_calls=100, fine=True)

            # Then, an ultra-fine pass if budget remains
            if call_count < budget - 1 and time_ok():
                # Even tighter offsets for final polish
                x_final = best_x.copy()
                f_final = best_f
                calls_used = 0
                ultra_offsets = [0.005, -0.005, 0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005]
                for _ in range(50):
                    if calls_used >= 100 or call_count >= budget - 1 or not time_ok():
                        break
                    improved = False
                    for i in range(dim):
                        if calls_used >= 100 or call_count >= budget - 1 or not time_ok():
                            break
                        base = x_final[i]
                        local_best = None
                        local_best_f = f_final
                        for off in ultra_offsets:
                            if calls_used >= 100 or call_count >= budget - 1 or not time_ok():
                                break
                            c = base + off
                            if not (-4.99 <= c <= 4.99):
                                continue
                            x_final[i] = c
                            cf = do_call(x_final)
                            calls_used += 1
                            if cf < local_best_f:
                                local_best_f = cf
                                local_best = c
                        if local_best is not None:
                            x_final[i] = local_best
                            f_final = local_best_f
                            improved = True
                        else:
                            x_final[i] = base
                    if not improved:
                        break
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final.copy()

        return best_x, best_f
