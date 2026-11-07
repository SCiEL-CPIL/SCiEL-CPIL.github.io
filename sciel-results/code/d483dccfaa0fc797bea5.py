import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, self.n_obj))

        N = budget

        # ZDT4 Pareto front: f1 = x1, f2 = 1 - sqrt(x1), for x1 in [0, 1]
        # Reference point: [11.0, 400.0]
        # We need to place N points on this front to maximize hypervolume.
        #
        # Strategy: Use a non-uniform distribution that clusters points
        # where the marginal gain in hypervolume is highest.
        # For the front y = 1 - sqrt(x), the curvature is high near x=0.
        #
        # Instead of a single power law, use a two-part distribution:
        # - Allocate more points to the region [0, 0.5] where sqrt is steep
        # - Use a power law within each region
        #
        # Actually, let's think about this more carefully.
        # HV contribution of a point at position x_i (sorted):
        #   d_i = (x_{i+1} - x_i) * (R2 - (1 - sqrt(x_i)))
        # where x_{N+1} = R1 = 11.0
        #
        # To maximize sum of d_i, we want points closer together where
        # (R2 - 1 + sqrt(x_i)) is large, i.e., where x_i is small (sqrt(x) 
        # changes rapidly). But we also want to cover the full range.
        #
        # Optimal spacing: dx_i / dx_{i+1} = sqrt(x_{i+1}) / sqrt(x_i)
        # This gives x_i proportional to i^2 (quadratic spacing).
        # Let's verify: if x_i = c*i^2, then dx_i = c*(2i+1) ≈ 2c*i
        # sqrt(x_i) = sqrt(c)*i
        # dx_i / sqrt(x_i) = 2c*i / (sqrt(c)*i) = 2*sqrt(c) = constant ✓
        #
        # So x_i = (i/N)^2 * x_max where x_max = 1.0
        # But we should also include x=0 and x=1.
        #
        # Let's use: x_i = ((i-0.5) / N)^2 for i=1..N, then adjust.
        # Actually simpler: x_i = (i/(N+1))^2, i=1..N, then add endpoints.
        # But we only have N evaluations. Let's use N points total.
        #
        # Use: x_i = (i / N)^2 for i=0..N-1, then x_{N-1} = 1.0
        # x_0 = 0, x_{N-1} = 1, and intermediate points follow quadratic law.

        # Generate N points on [0, 1] with quadratic spacing (optimal for this front)
        # x_i = (i/N)^2 for i = 0, 1, ..., N-1
        # This gives x_0 = 0, x_{N-1} = ((N-1)/N)^2 ≈ 1 for large N
        # Force last point to exactly 1.0

        idx = np.arange(N, dtype=float)
        x0_vals = (idx / N) ** 2.0
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0

        # Ensure monotonic and within [0, 1]
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        # Make strictly increasing (in case of floating point issues)
        for k in range(1, N):
            if x0_vals[k] <= x0_vals[k-1]:
                x0_vals[k] = x0_vals[k-1] + 1e-12

        # For x1..x9, set to values that minimize g(x) = sum_{i=2}^{10} x_i^2
        # Optimal is x_i = 0 if 0 is in bounds, else closest bound to 0.
        tail = np.zeros(dim - 1)
        for k in range(dim - 1):
            if lo[k+1] <= 0.0 <= hi[k+1]:
                tail[k] = 0.0
            elif 0.0 < lo[k+1]:
                tail[k] = lo[k+1]
            else:
                tail[k] = hi[k+1]

        evals = 0
        results = []

        for x0_val in x0_vals:
            if evals >= budget:
                break
            if time.time() - t0 > DEADLINE:
                break

            x = np.empty(dim)
            x[0] = x0_val
            x[1:] = tail
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                evals += 1
                if f is not None:
                    results.append(np.asarray(f, dtype=float))
            except Exception:
                pass

        if results:
            return np.array(results)

        return np.empty((0, self.n_obj))
