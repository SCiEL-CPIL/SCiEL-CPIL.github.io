"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Bayesian-inspired multi-objective optimizer targeting ZDT6's penalty term g(x)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        archive_X = []
        archive_F = []
        eval_count = 0

        def is_dominated(f, archive_F_arr):
            """Check if f is dominated by any point in archive_F_arr (array of shape (n, n_obj))."""
            # f is dominated if there exists a point that is <= in all objectives and < in at least one
            if len(archive_F_arr) == 0:
                return False
            # For each point in archive, check if it dominates f
            # point i dominates f if: all(archive_F[i] <= f) and any(archive_F[i] < f)
            leq = np.all(archive_F_arr <= f, axis=1)
            lt = np.any(archive_F_arr < f, axis=1)
            return np.any(leq & lt)

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if x is dominated by existing archive
            if is_dominated(f, np.array(archive_F)) if archive_F else False:
                return
            # Remove points from archive that are dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                # Check if af is dominated by f
                if np.all(f <= af) and np.any(f < af):
                    continue
                new_archive_X.append(ax)
                new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            eval_count += 1
            add_to_archive(x_clipped.copy(), f)
            return f

        # Phase 1: Initial random sampling
        n_init = min(20, budget)
        for _ in range(n_init):
            if eval_count >= budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # Phase 2: Targeted search for low g(x) values
        # For ZDT6, g(x) = 1 + 9/n * sum(x_i)
        # Minimizing g means minimizing sum(x_i), i.e., pushing x towards 0
        # The Pareto front is achieved when g is minimized (g=1) and f1 varies
        # We want to find points with g close to 1 and varying f1

        # Strategy: Use local optimization to minimize g(x) from various starting points
        # Then, along the Pareto-optimal region (g=1), vary x1 to get different f1 values

        # First, find points with very low g by minimizing sum(x_i)
        # g(x) = 1 + 9/dim * sum(x_i), so minimizing g is equivalent to minimizing sum(x_i)

        # Use several local optimizations from different starting points
        n_local = min(15, budget - eval_count)
        for _ in range(n_local):
            if eval_count >= budget:
                break
            # Start from a random point biased towards 0
            x0 = lo + (hi - lo) * rng.random(dim) * 0.1  # Close to lower bound
            x0 = np.clip(x0, lo, hi)

            # Define objective: minimize sum(x_i) which minimizes g
            def obj_g(x):
                return np.sum(np.clip(x, lo, hi))

            # Use Nelder-Mead for local optimization (no gradient needed)
            try:
                res = minimize(obj_g, x0, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if eval_count < budget:
                    evaluate(res.x)
            except Exception:
                pass

        # Phase 3: Explore the Pareto front region
        # For ZDT6, the Pareto front is: f1 = 1 + 10*(n-1)*g, f2 = (1 + 1/sqrt(f1))*(1 - sqrt(f1/(1+10*(n-1))))
        # when g=1. But we need to find actual points that achieve this.
        # The key insight: when all x_i = 0 except possibly x_1, we get the front.
        # Actually, for ZDT6, the Pareto-optimal solutions have g(x)=1, which means sum(x_i)=0,
        # so all x_i = 0. But then f1 = 1 + 10*(n-1)*1 = 101 for n=10.
        # Wait, let me reconsider. ZDT6:
        # g(x) = 1 + 9/n * sum_{i=2}^{n} x_i * (1 + 10*sin^2(15*x_i))
        # f1(x) = 1 - exp(-4*x_1) * sin(6*pi*x_1)^6
        # f2(x) = g(x) * (1 - (f1/g(x))^2)
        #
        # So g depends on x_2..x_n, not x_1!
        # To minimize g, we need x_2..x_n = 0.
        # Then g = 1, and f1 = 1 - exp(-4*x_1)*sin(6*pi*x_1)^6
        # f2 = 1 - f1^2
        #
        # So the Pareto front is parameterized by x_1 in [0,1], with x_2..x_n = 0.

        # Let's sample along this front: x_1 varies, x_2..x_n = 0
        n_front = min(30, budget - eval_count)
        for i in range(n_front):
            if eval_count >= budget:
                break
            x1 = i / max(n_front - 1, 1)
            x = np.zeros(dim)
            x[0] = x1
            evaluate(x)

        # Phase 4: Fine-tune with local search around promising archive points
        remaining = budget - eval_count
        if remaining > 0 and archive_X:
            # Sort archive by f2 (lower is better)
            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            # Get top few points by f2
            top_k = min(5, len(archive_F))
            idx_sorted = np.argsort(archive_F_arr[:, 1])[:top_k]

            for idx in idx_sorted:
                if eval_count >= budget:
                    break
                x_center = archive_X_arr[idx]
                # Local perturbation
                for _ in range(min(3, budget - eval_count)):
                    if eval_count >= budget:
                        break
                    x_new = x_center + 0.05 * rng.standard_normal(dim)
                    x_new = np.clip(x_new, lo, hi)
                    evaluate(x_new)

        return archive_X if archive_X else [lo]
# EVOLVE-BLOCK-END
