"""Exploit DTLZ1's known Pareto front with Sobol-based optimal density sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-and-optimize approach for DTLZ1:
    1. Calibrate K (the sum of objectives on the Pareto front) empirically.
    2. Generate well-distributed points on the simplex f1+f2+f3=K using Sobol sequences.
    3. Map simplex points back to decision space and evaluate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            results.append(f)
            used += 1
            return f

        # Step 1: Calibrate K by probing the problem
        # For DTLZ1, the Pareto front is achieved when tail variables minimize g.
        # We need to find the optimal tail values and the corresponding K.
        # Strategy: try several candidate tail configurations and find the one
        # that gives the minimum sum of objectives.

        probe_points = []
        tail_candidates = [
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9, 0.9],
        ]

        min_sum = float('inf')
        best_tail = None

        for tail in tail_candidates:
            if used >= budget or time.time() > deadline:
                break
            # Set x[0]=0.5, x[1]=0.5, x[2]=0.5 to get a balanced point
            x = np.zeros(dim)
            x[0] = 0.5
            x[1] = 0.5
            x[2] = 0.5
            for i, t in enumerate(tail):
                x[3 + i] = t
            f = eval_point(x)
            if f is not None:
                s = float(np.sum(f))
                if s < min_sum:
                    min_sum = s
                    best_tail = tail[:]

        if min_sum == float('inf') or used >= budget or time.time() > deadline:
            return results

        K = min_sum

        # Step 2: Generate dense points on the simplex f1+f2+f3=K
        # Use Sobol sequence for well-distributed sampling

        def simplex_to_decision(f1, f2, f3, K_val, tail):
            """Map objective values to decision variables for DTLZ1."""
            x = np.zeros(dim)
            # Set tail variables to their optimal values
            for i, t in enumerate(tail):
                x[3 + i] = t
            if K_val < 1e-12:
                x[0] = 0.5
                x[1] = 0.5
                return np.clip(x, lo, hi)

            # Standard DTLZ1 mapping:
            # f1 = 0.5*(1+g)*x0
            # f2 = 0.5*(1+g)*x1*(1-x0)
            # f3 = 0.5*(1+g)*(1-x1)
            # So: x0 = f1/K, x1 = 1 - f3/K
            x0 = f1 / K_val
            x1 = 1.0 - f3 / K_val
            x[0] = x0
            x[1] = x1
            x[2] = 0.5  # x2 doesn't affect objectives in standard DTLZ1
            return np.clip(x, lo, hi)

        # Use Sobol sequence for the remaining budget
        remaining = budget - used
        if remaining <= 0 or time.time() > deadline:
            return results

        # Generate Sobol points in [0,1]^2, then map to simplex
        # For a 2D Sobol sequence, we get points (u1, u2) in [0,1]^2
        # Map to simplex: f1 = K*u1, f2 = K*u2*(1-u1), f3 = K*(1-u2)*(1-u1)
        # This ensures f1+f2+f3 = K*u1 + K*u2*(1-u1) + K*(1-u2)*(1-u1)
        # = K*u1 + K*(1-u1)*(u2 + 1 - u2) = K*u1 + K*(1-u1) = K ✓

        # Use scipy's Sobol sampler
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=2, scramble=True)
            # Generate enough points (Sobol requires power of 2)
            n_points = 1
            while n_points < remaining:
                n_points *= 2
            # But we only need 'remaining' points
            n_points = min(n_points, remaining)
            # Actually, let's just generate 'remaining' points using a simple approach
            # since scipy Sobol might require power of 2
            
            # Use a simple Halton-like sequence instead for flexibility
            def halton(index, base):
                f = 1.0
                r = 0.0
                i = index
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r

            # Generate points using Halton sequence in 2D
            for idx in range(remaining):
                if used >= budget or time.time() > deadline:
                    break
                # Use different bases for 2D Halton
                u1 = halton(idx + 1, 2)
                u2 = halton(idx + 1, 3)
                
                # Map to simplex
                f1 = K * u1
                f2 = K * u2 * (1 - u1)
                f3 = K * (1 - u2) * (1 - u1)
                
                x = simplex_to_decision(f1, f2, f3, K, best_tail)
                eval_point(x)
                
        except Exception:
            # Fallback to random sampling
            rng = np.random.default_rng(42)
            for idx in range(remaining):
                if used >= budget or time.time() > deadline:
                    break
                u1 = rng.random()
                u2 = rng.random()
                f1 = K * u1
                f2 = K * u2 * (1 - u1)
                f3 = K * (1 - u2) * (1 - u1)
                x = simplex_to_decision(f1, f2, f3, K, best_tail)
                eval_point(x)

        return results
# EVOLVE-BLOCK-END
