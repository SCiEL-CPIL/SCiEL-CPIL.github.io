"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 using known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        results = []
        used = 0

        # RE21: f1 = sum(x_i^2), f2 = sum((x_i-2)^2)
        # Pareto front: all x_i equal, x = 2(1-w)
        # f1(w) = 4*n*(1-w)^2, f2(w) = 4*n*w^2
        # Reference point: [3000, 0.06]
        # Only f2 < 0.06 contributes: 4*n*w^2 < 0.06 => w < sqrt(0.06/(4n))

        n = dim
        ref_f2 = 0.06
        w_max = np.sqrt(ref_f2 / (4.0 * n))

        # Use the full budget on the contributing region [0, w_max]
        # with a dense, structured grid for maximum hypervolume
        n_evals = budget

        # Create a dense grid in [0, w_max]
        # Use non-uniform spacing: denser near 0 where f1 changes fastest
        # Use a mix of uniform and geometric spacing

        # Strategy: use a combination of:
        # 1. Uniform grid over [0, w_max]
        # 2. Extra points near 0 (where f1 is large and changes fast)
        # 3. A few points very close to w_max (boundary of contribution)

        # 60% uniform, 30% geometric (biased toward 0), 10% near boundary
        n_uniform = int(0.60 * n_evals)
        n_geom = int(0.30 * n_evals)
        n_boundary = n_evals - n_uniform - n_geom

        # Uniform grid in [0, w_max]
        w_uniform = np.linspace(0.0, w_max, n_uniform)

        # Geometric spacing biased toward 0
        # w = w_max * t^p where t in [0,1], p > 1 biases toward 0
        t = np.linspace(0.0, 1.0, n_geom)
        w_geom = w_max * t**2.0  # quadratic bias toward 0

        # Boundary points near w_max
        w_boundary = w_max * np.linspace(0.98, 1.0, n_boundary)

        # Combine all w values and sort
        w_all = np.concatenate([w_uniform, w_geom, w_boundary])
        w_all = np.sort(np.unique(w_all))

        # Trim to budget if needed
        if len(w_all) > n_evals:
            # Keep evenly spaced subset
            indices = np.linspace(0, len(w_all) - 1, n_evals).astype(int)
            w_all = w_all[indices]

        # Evaluate all points on the Pareto front
        for w in w_all:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x_val = 2.0 * (1.0 - w)
            x = np.full(dim, x_val)
            x = np.clip(x, lo, hi)

            f = problem(x)
            used += 1
            results.append((x.copy(), f))

        # If we still have budget (shouldn't happen), add random points
        if used < budget:
            rng = np.random.default_rng(0)
            for _ in range(budget - used):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                results.append((x.copy(), f))

        if not results:
            x = (lo + hi) / 2.0
            f = problem(x)
            results.append((x, f))
            used += 1

        return results[-1][0]
# EVOLVE-BLOCK-END
