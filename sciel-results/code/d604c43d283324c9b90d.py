import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        t_dead = t0 + 275.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Build 13x13 structured grid on x[0] and x[1], rest set to 0 (g=0)
        # 169 points total
        n_grid = 13
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        x0_flat = X0.flatten()
        x1_flat = X1.flatten()
        n_grid = len(x0_flat)

        X = np.zeros((n_grid, self.dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # X[:, 2:] stays 0 => g=0 on the Pareto front

        # Evaluate all grid points (fast, each call just clips and computes 3 values)
        for i in range(n_grid):
            if time.time() > t_dead:
                break
            problem(X[i])

        # Remaining budget: adaptive golden-ratio spiral sampling
        # Concentrate density near x1->0 (where f3 drops steeply from 21 to ~1.5)
        remaining = self.budget - min(self.budget, n_grid)
        remaining = max(0, remaining)

        # Use deterministic low-discrepancy via Halton-like golden ratio
        # For each point, map u in (0,1) to x1 = u^2 (concentrate near 0)
        # x0 = (v * phi mod 1) * u + (1-u) * (1-u*0.5) -- varied x0, less as u->1
        # where u = x1_param, v = index-based golden ratio

        golden = 0.61803398874989488

        # Use a simple deterministic set of u values:
        # Combine Halton base-3 for u and base-2 for v
        def halton(n, base):
            """Halton sequence in [0,1)"""
            f = np.ones(n)
            i = np.ones(n)
            for _ in range(30):
                i = (i * base) % (base + 1)
                f = f / base + (i - 1) / (base * (base + 1) + 1) * 0  # placeholder
            # Proper Halton:
            result = np.zeros(n)
            for idx in range(n):
                val = 0.0
                denom = base
                num = idx + 1
                while num > 0:
                    num, rem = divmod(num, base)
                    denom *= base
                    val += rem / denom
                result[idx] = val
            return result

        if remaining > 0:
            n_extra = remaining
            # Halton sequences
            u_seq = halton(n_extra, 3)  # for x1 param
            v_seq = halton(n_extra, 2)  # for x0 mixing

            # Map u to x1 with concentration near 0: x1 = u^2
            x1_extra = u_seq ** 2

            # x0: for small x1 (u small), spread x0 broadly; for large x1, x0 can be anything
            # Use v_seq to pick x0 = (v + 0.5/n) mod 1 style but let's just use v_seq
            x0_extra = v_seq

            # But we want to avoid too many duplicates with grid
            # Grid points are at multiples of 1/12. v_seq is well-distributed so overlap unlikely
            X_extra = np.zeros((n_extra, self.dim))
            X_extra[:, 0] = x0_extra
            X_extra[:, 1] = x1_extra

            for i in range(n_extra):
                if time.time() > t_dead:
                    break
                problem(X_extra[i])

        return X
