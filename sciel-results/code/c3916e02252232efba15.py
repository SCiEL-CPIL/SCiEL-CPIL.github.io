"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic lattice search + axis-aligned coordinate-wise line search
    for Schaffer's F7 (ill-conditioned, axis-aligned structure)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 285.0  # 15s safety margin before 300s wall-clock

        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                val = float(func(x))
            except Exception:
                return None
            calls[0] += 1
            if not np.isfinite(val):
                return None
            return val

        def time_ok():
            return time.time() < DEADLINE

        # ---- Phase 1: Coarse global lattice to find the best basin ----
        # 5^5 = 3125 points is too many; use a sparse lattice.
        # Try a grid of 3 points per axis: 3^5 = 243 points.
        # But 243 may be too many if budget is tight. Use adaptive.
        n_coarse = min(125, budget - 100)  # leave budget for refinement
        # 125 = 5^3, so 5 points in 3 axes, 1 in others? No, better:
        # Use a stratified sample: for each axis, pick points at -4, -2, 0, 2, 4
        # but that's 5^5=3125. Too many.
        # Instead: use a smaller grid. 3^5 = 243. If budget allows, use it.
        # Otherwise, use 2^5 = 32 or 125.

        # Strategy: use a grid of n_per_axis points where n_per_axis^dim <= n_coarse
        n_per_axis = max(2, int(round(n_coarse ** (1.0 / dim))))
        # Ensure n_per_axis^dim <= n_coarse
        while n_per_axis ** dim > n_coarse and n_per_axis > 2:
            n_per_axis -= 1

        # Generate grid points
        grid_coords = np.linspace(-4.0, 4.0, n_per_axis)
        grid_points = np.array(np.meshgrid(*([grid_coords] * dim))).T.reshape(-1, dim)

        xs = []
        fs = []
        for x in grid_points:
            if not time_ok() or calls[0] >= budget:
                break
            fx = safe_call(x)
            if fx is None:
                continue
            xs.append(x)
            fs.append(fx)

        if not fs:
            # Fallback
            x = np.zeros(dim)
            fx = safe_call(x)
            if fx is None:
                return None, None
            return x, fx

        fs = np.array(fs)
        xs = np.array(xs)
        best_idx = int(np.argmin(fs))
        x_best = xs[best_idx].copy()
        f_best = float(fs[best_idx])

        # ---- Phase 2: Dense local lattice around the best point ----
        # Use a small box around x_best, e.g., radius 1.0
        # Grid of 3 points per axis within [x_best - 1, x_best + 1]
        local_radius = 1.0
        n_local_per_axis = 3
        local_coords = np.linspace(-local_radius, local_radius, n_local_per_axis)
        local_points = np.array(np.meshgrid(*([local_coords] * dim))).T.reshape(-1, dim) + x_best
        local_points = np.clip(local_points, -5.0, 5.0)

        # Allocate remaining budget for local search
        remaining = budget - calls[0]
        n_local_max = min(len(local_points), max(10, remaining // 2))

        local_xs = []
        local_fs = []
        for x in local_points[:n_local_max]:
            if not time_ok() or calls[0] >= budget:
                break
            fx = safe_call(x)
            if fx is None:
                continue
            local_xs.append(x)
            local_fs.append(fx)

        if local_fs:
            local_fs = np.array(local_fs)
            local_xs = np.array(local_xs)
            local_best_idx = int(np.argmin(local_fs))
            if local_fs[local_best_idx] < f_best:
                f_best = float(local_fs[local_best_idx])
                x_best = local_xs[local_best_idx].copy()

        # ---- Phase 3: Axis-aligned coordinate-wise line search ----
        # For each coordinate, perform a 1-D search along that axis
        # Use golden-section search for efficiency

        def golden_section_search(func_1d, a, b, tol=1e-8, max_iter=50):
            """Golden-section search for minimum of func_1d on [a, b]."""
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = func_1d(c)
            fd = func_1d(d)
            for _ in range(max_iter):
                if not time_ok() or calls[0] >= budget:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = func_1d(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = func_1d(d)
                if abs(b - a) < tol:
                    break
            return (a + b) / 2.0

        # For each coordinate, search along that axis
        for coord in range(dim):
            if not time_ok() or calls[0] >= budget:
                break

            # Determine search range for this coordinate
            # Use a small range around current best
            search_radius = 0.5
            a = max(-5.0, x_best[coord] - search_radius)
            b = min(5.0, x_best[coord] + search_radius)

            if b - a < 1e-6:
                continue

            # Define 1-D function: fix all other coordinates, vary coord
            def func_1d(t):
                x_temp = x_best.copy()
                x_temp[coord] = t
                return safe_call(x_temp)

            # Perform golden-section search
            t_opt = golden_section_search(func_1d, a, b)

            # Evaluate at the optimal point
            x_new = x_best.copy()
            x_new[coord] = t_opt
            fx_new = safe_call(x_new)

            if fx_new is not None and fx_new < f_best:
                f_best = fx_new
                x_best = x_new

        # ---- Phase 4: Additional coordinate-wise refinement passes ----
        # Repeat the coordinate-wise search with smaller radii for further refinement
        for pass_num in range(3):
            if not time_ok() or calls[0] >= budget:
                break
            search_radius = 0.1 / (pass_num + 1)
            for coord in range(dim):
                if not time_ok() or calls[0] >= budget:
                    break

                a = max(-5.0, x_best[coord] - search_radius)
                b = min(5.0, x_best[coord] + search_radius)

                if b - a < 1e-8:
                    continue

                def func_1d(t, coord=coord):
                    x_temp = x_best.copy()
                    x_temp[coord] = t
                    return safe_call(x_temp)

                t_opt = golden_section_search(func_1d, a, b)

                x_new = x_best.copy()
                x_new[coord] = t_opt
                fx_new = safe_call(x_new)

                if fx_new is not None and fx_new < f_best:
                    f_best = fx_new
                    x_best = x_new

        return x_best, f_best
# EVOLVE-BLOCK-END
