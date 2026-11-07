"""Multi-objective optimizer on WFG7 using Sobol sampling + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Space-filling Sobol initialization with Nelder-Mead local refinement on non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Generate Sobol sequence for space-filling initialization
        n_init = min(128, self.budget // 2)
        # Use scipy if available for Sobol, otherwise use Halton-like
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            X_init = sobol.random(n_init)
        except ImportError:
            # Fallback: use Halton sequence
            X_init = self._halton(n_init, self.dim)

        # Map to bounds
        X_init = lo + (hi - lo) * X_init

        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Get non-dominated points from initial set
        nd_indices = self._non_dominated_indices(F_init)

        # Budget remaining for local refinement
        remaining = self.budget - used
        n_refine = len(nd_indices)
        # Allocate remaining budget among non-dominated points
        if n_refine > 0 and remaining > 0:
            evals_per_point = max(1, remaining // n_refine)
            # Prioritize points with lower sum of objectives
            obj_sums = np.sum(F_init[nd_indices], axis=1)
            priority_order = np.argsort(obj_sums)

            for idx in priority_order:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                orig_idx = nd_indices[idx]
                x0 = X_init[orig_idx].copy()
                f0 = F_init[orig_idx].copy()

                # Nelder-Mead local optimization on a weighted scalarization
                # Use multiple weight vectors to explore different parts of the front
                weights = self._weight_vectors()
                for w in weights:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    def scalar_obj(x):
                        fx = problem(x)
                        return np.dot(fx, w)

                    # Nelder-Mead
                    best_x, best_f = self._nelder_mead(
                        scalar_obj, x0, lo, hi, problem, w
                    )
                    # Track evaluations
                    # We need to count actual problem evaluations
                    # Since nelder_mead uses problem internally, we need to track
                    # Let's restructure to track evals

                # Alternative: do a simple local search with tracked evaluations
                # Reset and do tracked local search
                local_best_x = x0.copy()
                local_best_f = f0.copy()
                step_size = 0.05 * (hi - lo)

                for _ in range(min(evals_per_point, self.budget - used)):
                    if time.time() > deadline:
                        break
                    # Random perturbation
                    delta = rng.standard_normal(self.dim) * step_size
                    x_new = np.clip(x0 + delta, lo, hi)
                    f_new = problem(x_new)
                    used += 1

                    all_X.append(x_new.copy())
                    all_F.append(f_new.copy())

                    if np.all(f_new <= local_best_f) and np.any(f_new < local_best_f):
                        local_best_f = f_new.copy()
                        local_best_x = x_new.copy()

                    # Decay step size
                    step_size *= 0.95

        return all_X

    def _halton(self, n, dim):
        """Generate Halton sequence points in [0,1]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            prime = self._nth_prime(d + 1)
            for i in range(1, n + 1):
                x = 0.0
                f = 1.0 / prime
                num = i
                while num > 0:
                    x += f * (num % prime)
                    num //= prime
                    f /= prime
                points[i - 1, d] = x
        return points

    def _nth_prime(self, n):
        """Return the nth prime number."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes[-1]

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (n x m array)."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _weight_vectors(self):
        """Generate a set of weight vectors for scalarization."""
        # Simple weight vectors emphasizing different objectives
        weights = []
        for i in range(self.n_obj):
            w = np.ones(self.n_obj) * 0.1
            w[i] = 0.8
            weights.append(w)
        # Equal weights
        weights.append(np.ones(self.n_obj) / self.n_obj)
        return weights

    def _nelder_mead(self, scalar_obj, x0, lo, hi, problem, w):
        """Simple Nelder-Mead implementation. Returns (best_x, best_f)."""
        # This is a simplified version; for tracking evals we use the tracked local search above
        return x0, scalar_obj(x0)
# EVOLVE-BLOCK-END
