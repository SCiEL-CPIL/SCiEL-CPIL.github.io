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

        # Evaluate initial points
        for pt in initial_points:
            if eval_count >= budget or time.time() > deadline:
                break
            val = counting_func(pt)

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
        # Sort initial evaluations by value
        # We need to track which points gave which values
        # Re-evaluate is wasteful, so let's collect during evaluation

        # Actually, let's redo: collect (value, point) pairs
        # But we already evaluated. Let's just use the best few from initial.
        # Since we don't have the full list, let's do a second pass:
        # Use the best point and a few neighbors.

        # Better approach: run CMA-ES from the best point and a few others.
        # Select top candidates by re-examining: we'll just use the best and
        # generate neighbors around it.

        remaining = budget - eval_count
        if remaining > 50 and best_x is not None:
            # Run CMA-ES from the best point
            # Also try a few other promising points

            # Generate diverse starting points: best + perturbations
            start_points = [best_x.copy()]

            # Add points from the initial set that were good
            # Since we didn't store them, let's just add perturbations
            for scale in [0.5, 1.0, 2.0]:
                for coord in range(dim):
                    for direction in [-1, 1]:
                        pt = best_x.copy()
                        pt[coord] += direction * scale
                        pt = np.clip(pt, -5, 5)
                        start_points.append(pt)

            # Deduplicate
            unique_starts = []
            for pt in start_points:
                is_dup = False
                for u in unique_starts:
                    if np.linalg.norm(pt - u) < 0.1:
                        is_dup = True
                        break
                if not is_dup:
                    unique_starts.append(pt)

            # Limit to top 3 starting points
            unique_starts = unique_starts[:3]

            # Allocate budget: give more to the best start
            n_starts = len(unique_starts)
            base_alloc = max(30, (remaining - n_starts * 10) // n_starts)

            for i, x_start in enumerate(unique_starts):
                if eval_count >= budget or time.time() > deadline:
                    break

                local_budget = min(base_alloc, budget - eval_count)
                if local_budget < 20:
                    break

                popsize = 15
                try:
                    res = scipy.optimize.minimize(
                        counting_func,
                        x_start,
                        method='CMA',
                        options={
                            'maxiter': local_budget // popsize + 5,
                            'maxfev': local_budget,
                            'popsize': popsize,
                            'sigma0': 0.5,
                            'bounds': [(-5.0, 5.0)] * dim,
                            'verbose': 0,
                            'tol': 1e-10
                        }
                    )
                except Exception:
                    continue

        # --- Phase 3: Final polish with Nelder-Mead ---
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # --- Phase 4: If budget remains, do a fine local search ---
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            # Coordinate-wise line search
            for coord in range(dim):
                if eval_count >= budget or time.time() > deadline:
                    break
                # Try small steps in each direction
                for step in [0.01, -0.01, 0.005, -0.005, 0.1, -0.1]:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    pt = best_x.copy()
                    pt[coord] += step
                    pt = np.clip(pt, -5, 5)
                    val = counting_func(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
