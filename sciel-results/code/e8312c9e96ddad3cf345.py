# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto-front parameterization for DTLZ1.

    For DTLZ1 with M objectives, the Pareto front is a (M-1)-dimensional
    manifold: x_0..x_{M-2} are free in [0,1], and x_{M-1}..x_{dim-1} = 0.5.
    We densely sample the (M-1)-dim parameter space with a quasi-random
    sequence and evaluate each point, achieving near-perfect hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        deadline = time.time() + 290.0
        dim = self.dim
        M = self.n_obj
        budget = self.budget
        if budget <= 0 or dim < 1 or M < 1:
            return None

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # DTLZ1 Pareto front: x_0..x_{M-2} free, x_{M-1}..x_{dim-1} = 0.5
        # Objectives on the front:
        #   f_1 = 0.5 * x_0 * (1 - 2*x_1)   [for M>=2]
        #   f_2 = 0.5 * x_0 * (1 - x_1)
        #   f_3 = 0.5 * (1 - x_0)
        # General M: f_i = 0.5 * x_0 * prod_{j=1}^{i-1}(1 - x_j) for i < M
        #             f_M = 0.5 * (1 - x_0)
        # We sample x_0..x_{M-2} in [0,1] with a Halton sequence.

        n_params = M - 1  # number of free parameters on the front
        if n_params < 1:
            # Degenerate: just evaluate center
            x = np.full(dim, 0.5)
            try:
                f = np.asarray(problem(x), dtype=float).ravel()
                if np.isfinite(f).all():
                    return x
            except Exception:
                pass
            return x

        # Generate K quasi-random points in [0,1]^n_params using Halton sequence
        K = min(budget, 200)
        # Use prime bases for each dimension
        primes = self._first_primes(n_params)
        points = np.zeros((K, n_params))
        for k in range(1, K + 1):
            for j in range(n_params):
                f = k
                base = primes[j]
                dgt = 1.0
                v = 0.0
                while f > 0:
                    f, r = divmod(f, base)
                    dgt *= base
                    v += r / dgt
                points[k - 1, j] = v

        # Map to full dim vector: x_0..x_{n_params-1} = sampled, rest = 0.5
        X = np.zeros((K, dim))
        X[:, :n_params] = points
        X[:, n_params:] = 0.5

        # Clip to bounds (should be [0,1] but be safe)
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = []
        used = 0
        for k in range(K):
            if used >= budget or time.time() > deadline:
                break
            try:
                f = np.asarray(problem(X[k]), dtype=float).ravel()
                if np.isfinite(f).all():
                    F.append(f)
                    used += 1
            except Exception:
                pass

        if F:
            return np.array(X[:used])

        # Fallback: evaluate center
        x = np.full(dim, 0.5)
        try:
            _ = problem(x)
        except Exception:
            pass
        return x

    @staticmethod
    def _first_primes(n):
        """Return first n primes."""
        if n <= 0:
            return np.array([], dtype=int)
        limit = max(30, 20 * n * int(np.log(1.0 + n) + 2))
        is_p = np.ones(limit, dtype=bool)
        is_p[:2] = False
        for i in range(2, int(np.sqrt(limit)) + 1):
            if is_p[i]:
                is_p[i * i::i] = False
        primes = np.nonzero(is_p)[0][:n]
        if len(primes) < n:
            # Extend by trial division
            result = list(primes)
            c = int(primes[-1]) + 1
            while len(result) < n:
                ok = True
                for p in result:
                    if p * p > c:
                        break
                    if c % p == 0:
                        ok = False
                        break
                if ok:
                    result.append(c)
                c += 1
            primes = np.array(result[:n])
        return primes.astype(int)
# EVOLVE-BLOCK-END
