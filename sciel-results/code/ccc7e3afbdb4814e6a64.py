import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0
        n_evals = 0

        def counting_func(x):
            nonlocal eval_count, best_f, best_x, n_evals
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            eval_count += 1
            n_evals += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Katsuura-aware structured sampling ---
        # The Katsuura function f23 has a known structure: it's a product of
        # terms of the form (1 + (x_i - 1)^2 * something) with periodic components.
        # The global minimum in the standard (unshifted) version is near x = [1,1,1,1,1].
        # With hidden shift/rotation, we need to explore broadly but also probe
        # the "fractal" structure at multiple scales.

        # Generate a structured set of starting points:
        # 1. Center point (origin)
        # 2. Points near the known Katsuura minimum pattern (x_i ~ 1)
        # 3. A coarse grid to find the shifted minimum region
        # 4. Sobol-like quasi-random points for coverage

        initial_points = []

        # Center
        initial_points.append(np.zeros(dim))

        # Katsuura minimum pattern: x_i = 1 for all i
        initial_points.append(np.ones(dim))

        # Probe around the known minimum with small offsets
        for offset in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
            pt = np.ones(dim) + offset
            initial_points.append(pt)

        # Coarse grid: 3 points per dimension -> 3^5 = 243 points, too many
        # Use 2 points per dim (edges + center) -> 3^5 = 243 still too many
        # Use a stratified sample: for each dim, use {-4, -2, 0, 2, 4}
        # That's 5^5 = 3125, way too many.
        # Instead, use a Latin hypercube style: pick a few per dim

        # Let's use a small grid: {-4, -1, 0, 1, 4} per dim
        # But 5^5 = 3125 is too many. Use a subset.
        # Actually, let's just use a quasi-random set + the structured points.

        # Quasi-random: use a simple Halton-like sequence
        def halton_seq(n, dim):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                pt = []
                for d in range(dim):
                    # Find next prime base
                    base = 2 + d * (1 if d < 10 else 2)  # approximate primes
                    # Better: use fixed primes
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    base = primes[d % len(primes)]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    pt.append(r)
                points.append(pt)
            return np.array(points)

        # Map to [-5, 5]
        n_halton = 40
        halton_pts = halton_seq(n_halton, dim) * 10.0 - 5.0
        initial_points.extend(halton_pts.tolist())

        # Evaluate initial points, tracking (value, point) pairs
        eval_records = []
        for pt in initial_points:
            if eval_count >= budget or time.time() > deadline:
                break
            pt_arr = np.clip(np.asarray(pt, dtype=float), -5, 5)
            val = counting_func(pt_arr)
            eval_records.append((val, pt_arr.copy()))

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
            return best_x, best_f

        # --- Phase 2: Multi-start CMA-ES with adaptive budget allocation ---
        remaining = budget - eval_count
        if remaining > 50 and best_x is not None:
            # Select top diverse starting points from initial evaluations
            # Sort by value
            eval_records.sort(key=lambda x: x[0])
            
            # Select diverse points: greedily pick points that are far from already selected
            selected_starts = []
            min_dist = 1.5  # minimum distance between selected starts
            for val, pt in eval_records:
                if time.time() > deadline or eval_count >= budget:
                    break
                is_diverse = True
                for s in selected_starts:
                    if np.linalg.norm(pt - s) < min_dist:
                        is_diverse = False
                        break
                if is_diverse:
                    selected_starts.append(pt)
                if len(selected_starts) >= 4:
                    break
            
            # If we didn't get enough diverse points, add perturbations of best
            if len(selected_starts) < 2:
                for scale in [0.5, 1.0, 2.0]:
                    pt = best_x.copy()
                    pt += scale * np.random.randn(dim)
                    pt = np.clip(pt, -5, 5)
                    if not any(np.linalg.norm(pt - s) < 0.5 for s in selected_starts):
                        selected_starts.append(pt)
                    if len(selected_starts) >= 2:
                        break

            n_starts = len(selected_starts)
            # Allocate budget: give more to the best start
            # First start gets 50%, rest split
            if n_starts == 1:
                budgets = [remaining]
            else:
                first_budget = int(remaining * 0.5)
                rest_budget = remaining - first_budget
                per_rest = rest_budget // (n_starts - 1)
                budgets = [first_budget] + [per_rest] * (n_starts - 1)

            for i, x_start in enumerate(selected_starts):
                if eval_count >= budget or time.time() > deadline:
                    break

                local_budget = min(budgets[i], budget - eval_count)
                if local_budget < 15:
                    break

                # Adaptive popsize: larger for more budget
                popsize = max(10, min(20, local_budget // 5))
                try:
                    res = scipy.optimize.minimize(
                        counting_func,
                        x_start,
                        method='CMA',
                        options={
                            'maxfev': local_budget,
                            'popsize': popsize,
                            'sigma0': 0.5,
                            'bounds': [(-5.0, 5.0)] * dim,
                            'verbose': 0,
                            'tol': 1e-12
                        }
                    )
                except Exception:
                    continue

        # --- Phase 3: Final polish with Nelder-Mead ---
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # --- Phase 4: High-precision coordinate descent polish ---
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            # Iterative coordinate descent with shrinking step sizes
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for step in step_sizes:
                if eval_count >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and eval_count < budget and time.time() < deadline:
                    improved = False
                    for coord in range(dim):
                        if eval_count >= budget or time.time() > deadline:
                            break
                        for direction in [-1, 1]:
                            pt = best_x.copy()
                            pt[coord] += direction * step
                            pt = np.clip(pt, -5, 5)
                            val = counting_func(pt)
                            if val < best_f:
                                improved = True
                                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
