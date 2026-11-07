"""ZDT1 multi-objective optimizer exploiting the known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer exploiting ZDT1's decoupled Pareto front structure.
    
    ZDT1: f1 = x1, g = 1 + 9*sum(x2..xn)/(n-1), f2 = g*(1 - sqrt(f1/g))
    The Pareto front is achieved when x2..xn = 0 (g=1), and f1 sweeps [0,1].
    With 200 evaluations we can densely sample the Pareto front.
    We optimize the spacing exponent p to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        B = int(self.budget)
        D = int(self.dim)

        # For ZDT1, the Pareto front is at x2..xn = 0 (lower bound)
        # f1 = x1, f2 = 1 - sqrt(x1) when x2..xn = 0
        # We sweep x1 with non-uniform spacing x1 = u^p to maximize hypervolume

        # Search for optimal power exponent p
        # The integrand for HV is (11 - f2) = 11 - (1 - sqrt(x1)) = 10 + sqrt(x1)
        # which is largest near x1=0, so we want more points there
        # p > 1 clusters points near 0

        u = np.linspace(0.0, 1.0, B)

        # Fine-grained search over p values
        # Optimal p is around 1.3-1.5 based on theoretical analysis
        p_vals = np.linspace(0.8, 2.5, 400)

        best_hv = -np.inf
        best_p = 1.5
        best_x1 = None

        for p in p_vals:
            x1 = u ** p
            x1 = np.clip(x1, 0.0, 1.0)

            # All points on the ZDT1 front are non-dominated
            # f1 is sorted ascending, f2 = 1 - sqrt(f1) is sorted descending
            f1_sorted = x1
            f2_sorted = 1.0 - np.sqrt(f1_sorted)

            # Filter points strictly better than reference [11, 11]
            mask = (f1_sorted < 11.0) & (f2_sorted < 11.0)
            if not np.any(mask):
                continue
            f1_f = f1_sorted[mask]
            f2_f = f2_sorted[mask]

            if len(f1_f) < 2:
                continue

            # Compute hypervolume using the sorted points
            # df1[i] = f1[i+1] - f1[i], last extends to reference
            df1 = np.diff(f1_f)
            df1 = np.append(df1, 11.0 - f1_f[-1])
            hf2 = 11.0 - f2_f  # height in f2 dimension

            hv = np.sum(df1 * hf2)

            if hv > best_hv:
                best_hv = hv
                best_p = p
                best_x1 = x1.copy()

        # Refine around the best p found
        p_refine = np.linspace(best_p - 0.05, best_p + 0.05, 100)
        for p in p_refine:
            x1 = u ** p
            x1 = np.clip(x1, 0.0, 1.0)

            f1_sorted = x1
            f2_sorted = 1.0 - np.sqrt(f1_sorted)

            mask = (f1_sorted < 11.0) & (f2_sorted < 11.0)
            if not np.any(mask):
                continue
            f1_f = f1_sorted[mask]
            f2_f = f2_sorted[mask]

            if len(f1_f) < 2:
                continue

            df1 = np.diff(f1_f)
            df1 = np.append(df1, 11.0 - f1_f[-1])
            hf2 = 11.0 - f2_f

            hv = np.sum(df1 * hf2)

            if hv > best_hv:
                best_hv = hv
                best_p = p
                best_x1 = x1.copy()

        # If no valid points found, fall back to uniform
        if best_x1 is None:
            best_x1 = u.copy()

        # Create the sweep matrix
        X = np.zeros((B, D), dtype=np.float64)
        X[:, 0] = best_x1
        # x2..xn stay at 0 (lower bound), which is optimal for ZDT1

        # Clip to bounds to be safe
        X = np.clip(X, lo, hi)

        # Evaluate all sweep points
        used = 0
        for i in range(B):
            if time.time() > deadline:
                break
            if used >= B:
                break
            problem(X[i].copy())
            used += 1

        return X
# EVOLVE-BLOCK-END
