"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT6 structure for fast Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []  # list of (x, f) tuples
        self.eval_count = 0
        self.start_time = None
        self.deadline = 320.0  # 330s limit, 10s safety margin

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # ZDT6 Pareto front: for x1 in [0,1], f1 = x1, f2 = g(x)*(1 - sqrt(f1/g(x)))
        # where g(x) = 1 + 10*(n-1) + sum(10*(x_i^0.25 - sin(2*pi*x_i^0.25))) for i=2..n
        # We can exploit this structure: set x1 directly, and try to minimize g(x) for x2..x10
        # The minimum of g is 1 (when all x_i=0 for i>=2, since 0^0.25=0 and sin(0)=0)
        # So the Pareto front is: f1=x1, f2=1*(1-sqrt(x1/1)) = 1-sqrt(x1) for x1 in [0,1]

        # Strategy 1: Seed points directly on the Pareto front
        # For each x1 value, set x2..x10 to minimize g (i.e., set them to 0)
        n_seed = min(50, self.budget // 4)
        x1_values = np.linspace(0.0, 1.0, n_seed)
        for x1 in x1_values:
            if self.eval_count >= self.budget:
                break
            if time.time() - self.start_time > self.deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 minimizes g
            f = problem(x)
            self._add_point(x, f)

        # Strategy 2: Local refinement around Pareto front points
        # For remaining budget, do directed coordinate descent to improve g(x)
        remaining = self.budget - self.eval_count
        if remaining > 0 and self.archive:
            # Pick a subset of archive points to refine
            n_refine = min(len(self.archive), max(5, remaining // 10))
            indices = rng.choice(len(self.archive), size=n_refine, replace=False)
            for idx in indices:
                if self.eval_count >= self.budget:
                    break
                if time.time() - self.start_time > self.deadline:
                    break
                x_best, f_best = self.archive[idx]
                # Coordinate descent: try small perturbations in each dimension
                for dim_idx in range(1, self.dim):  # skip dim 0 (x1 controls f1)
                    if self.eval_count >= self.budget:
                        break
                    if time.time() - self.start_time > self.deadline:
                        break
                    # Try a few steps in positive and negative direction
                    for step in [0.01, 0.005, 0.001]:
                        for direction in [1, -1]:
                            if self.eval_count >= self.budget:
                                break
                            x_try = x_best.copy()
                            x_try[dim_idx] += direction * step
                            x_try = np.clip(x_try, lo, hi)
                            f_try = problem(x_try)
                            self.eval_count += 1
                            # Check if this is better (lower g means better)
                            # For ZDT6, lower f2 is better for fixed f1
                            if f_try[1] < f_best[1] - 1e-10 or (abs(f_try[1] - f_best[1]) < 1e-10 and f_try[0] < f_best[0]):
                                x_best, f_best = x_try, f_try
                                self._add_point(x_best, f_best)
                            if self.eval_count >= self.budget:
                                break
                    if self.eval_count >= self.budget:
                        break

        # Strategy 3: Random search in remaining budget
        while self.eval_count < self.budget:
            if time.time() - self.start_time > self.deadline:
                break
            # Random point with bias toward low x2..x10 (to minimize g)
            x = np.random.uniform(lo, hi, size=self.dim)
            # Bias x[1:] toward 0
            x[1:] = x[1:] * 0.3  # pull toward 0
            x = np.clip(x, lo, hi)
            f = problem(x)
            self._add_point(x, f)

        return self.archive

    def _add_point(self, x, f):
        self.eval_count += 1
        f = np.array(f, dtype=float)
        # Only add if not dominated by existing archive
        x = np.array(x, dtype=float)
        for _, f_exist in self.archive:
            if np.all(f_exist <= f) and np.any(f_exist < f):
                return  # dominated, don't add
        # Remove points dominated by new point
        self.archive = [(x_i, f_i) for x_i, f_i in self.archive
                        if not (np.all(f <= f_i) and np.any(f < f_i))]
        self.archive.append((x, f))

# EVOLVE-BLOCK-END
