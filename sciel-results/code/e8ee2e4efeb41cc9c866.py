"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart optimizer: global sampling + quadratic jump + Powell ridge refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        def track(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---- Powell direction-based line search (ridge-friendly refinement) ----
        def powell_refine(x0, f0):
            nonlocal best_x, best_f, calls_made
            n = dim
            x = x0.copy()
            f = f0
            track(x, f)

            # Initial directions = coordinate axes
            dirs = [np.eye(n, dtype=float)[i].copy() for i in range(n)]
            # Per-direction best point and value
            best_pts = [x.copy() for _ in range(n)]
            best_vals = [f] * n
            max_iters = 25

            for _ in range(max_iters):
                if calls_made >= budget or time.time() > deadline:
                    break
                old_f = f
                # Line minimization along each direction (golden section)
                for i in range(n):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    d = dirs[i]
                    a = -5.0
                    b = 5.0
                    # Ensure current point is inside [a, b]
                    if x[i] - a < 1e-9 or b - x[i] < 1e-9:
                        continue
                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                    c = b - gr * (b - a)
                    dpt = a + gr * (b - a)
                    fc = safe_call(np.clip(x + c * d, -5, 5)); calls_made += 1
                    fd = safe_call(np.clip(x + dpt * d, -5, 5)); calls_made += 1
                    track(x + c * d, fc); track(x + dpt * d, fd)
                    iters = 0
                    while (b - a) > 1e-7 and iters < 30:
                        iters += 1
                        if calls_made >= budget or time.time() > deadline:
                            break
                        if fc < fd:
                            b = dpt
                            dpt = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = safe_call(np.clip(x + c * d, -5, 5)); calls_made += 1
                            track(x + c * d, fc)
                        else:
                            a = c
                            c = dpt
                            fc = fd
                            dpt = a + gr * (b - a)
                            fd = safe_call(np.clip(x + dpt * d, -5, 5)); calls_made += 1
                            track(x + dpt * d, fd)
                    # Best along this direction
                    t = c if fc < fd else dpt
                    fv = fc if fc < fd else fd
                    new_x = np.clip(x + t * d, -5, 5)
                    new_f = fv
                    # Update global best point
                    if new_f < f:
                        x = new_x
                        f = new_f
                        track(x, f)
                    # Update per-direction best
                    if new_f < best_vals[i]:
                        best_pts[i] = new_x.copy()
                        best_vals[i] = new_f
                    else:
                        # keep direction but reset reference to best
                        new_x = best_pts[i]
                        new_f = best_vals[i]
                    # Move x to best point found along this axis
                    x = new_x
                    f = new_f

                if f >= old_f - 1e-14:
                    # No progress this round; stop
                    break

                # Reflection: new direction = current - previous best point
                prev_best = min(range(n), key=lambda i: best_vals[i])
                new_dir = x - best_pts[prev_best]
                if np.linalg.norm(new_dir) > 1e-12:
                    dirs[prev_best] = new_dir / np.linalg.norm(new_dir)
                    # Reset that direction's best to current
                    best_pts[prev_best] = x.copy()
                    best_vals[prev_best] = f

            return x, f

        # ---- Quadratic fit to jump toward predicted minimum ----
        def run_quadratic(x0, f0, radius):
            nonlocal best_x, best_f, calls_made
            n = dim
            if calls_made + 2 * n + 1 >= budget or time.time() > deadline:
                return x0, f0
            x = x0.copy()
            fx = f0
            sigma = radius

            points = [x.copy()]
            for i in range(n):
                e = np.zeros(n)
                e[i] = sigma
                points.append(np.clip(x + e, -5, 5))
                points.append(np.clip(x - e, -5, 5))

            values = [fx]
            for p in points[1:]:
                val = safe_call(p)
                calls_made += 1
                track(p, val)
                values.append(val)

            values = np.array(values)
            pts = np.array(points)
            X = pts - x0

            def make_features(z):
                feat = [1.0]
                feat.extend(z)
                for i in range(n):
                    for j in range(i, n):
                        feat.append(z[i] * z[j])
                return feat

            F = np.array([make_features(z) for z in X]).T  # (n_params, n_pts)
            try:
                coef, _, _, _ = np.linalg.lstsq(F, values, rcond=None)
            except np.linalg.LinAlgError:
                return x0, f0

            c = coef[0]
            g = coef[1:1 + n].copy()
            h = np.zeros((n, n))
            idx = 1 + n
            for i in range(n):
                for j in range(i, n):
                    val = coef[idx]
                    idx += 1
                    if i == j:
                        h[i, i] = val
                    else:
                        h[i, j] = val
                        h[j, i] = val

            try:
                eigvals, eigvecs = np.linalg.eigh(h)
                eigvals_clipped = np.maximum(eigvals, 1e-6)
                h_pos = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
                dx = np.linalg.solve(h_pos, -g)
            except np.linalg.LinAlgError:
                return x0, f0

            step_norm = np.linalg.norm(dx)
            max_step = 3.0 * sigma
            if step_norm > max_step:
                dx = dx * (max_step / step_norm)

            new_x = np.clip(x + dx, -5.0, 5.0)
            new_f = safe_call(new_x)
            calls_made += 1
            track(new_x, new_f)
            if new_f < fx:
                fx = new_f
                x = new_x
            return x, fx

        # ---- Main loop ----
        rng_master = np.random.default_rng(42)

        # Global sampling
        n_init = min(15, max(5, budget // 12))
        candidates = []
        for _ in range(n_init):
            x = rng_master.uniform(-4, 4, dim)
            fx = safe_call(x)
            calls_made += 1
            candidates.append((fx, x.copy()))
            track(x, fx)

        candidates.sort(key=lambda t: t[0])

        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Split budget among top restart points
        n_active = min(3, len(candidates))
        budgets_per_restart = [remaining // n_active] * n_active
        budgets_per_restart[-1] += remaining - sum(budgets_per_restart)

        for i in range(n_active):
            if time.time() > deadline or calls_made >= budget:
                break
            x_start = candidates[i][1]
            f_start = candidates[i][0]
            b = budgets_per_restart[i]
            if b <= 0:
                break

            # Adaptive quadratic radius: shrink as we get closer to convergence
            gap = f_start - best_f
            radius = 1.0
            if gap < 1e-3:
                radius = 0.3
            elif gap < 1e-2:
                radius = 0.6

            # Quadratic jump first (if budget allows), then Powell refinement
            if b > 2 * dim + 5:
                xq, fq = run_quadratic(x_start, f_start, radius)
                # Powell refinement from the quadratic result
                powell_refine(xq, fq)
            else:
                powell_refine(x_start, f_start)

        return best_x, best_f
# EVOLVE-BLOCK-END
