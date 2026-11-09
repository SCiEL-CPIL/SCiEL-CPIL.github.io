# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_step(self, func, x, sigma, C, pc, ps, weights, mu,
                     n, pop_size, rng, time_limit):
        """One CMA-ES iteration: sample pop, update, return new state + best."""
        N = n
        # Sample population
        # Cholesky of C
        L = np.linalg.cholesky(C + 1e-12 * np.eye(N))
        Y = []
        for i in range(pop_size):
            z = rng.standard_normal(N)
            Y.append(x + sigma * (L @ z))
        Y = np.array(Y)
        # Evaluate
        F = np.empty(pop_size)
        for i in range(pop_size):
            F[i] = func(Y[i])
        # Sort by fitness
        order = np.argsort(F)
        Y = Y[order]
        F = F[order]
        # New mean (weighted recombination)
        x_new = np.zeros(N)
        for i in range(mu):
            x_new += weights[i] * Y[i]
        # Best in this population
        best_x = Y[0].copy()
        best_f = F[0]
        # Check time
        if time.time() > time_limit:
            return x_new, sigma, C, pc, ps, best_x, best_f, True
        # Update step size control variable
        # ps = (1 - mu_sigma) * ps + mu_sigma / sigma * (L^{-1} * (x_new - x))
        # mu_sigma = sum(weights[:mu])^2 / sum(weights[:mu]^2)
        mu_sigma = np.sum(weights[:mu] ** 2) / np.sum(weights[:mu] ** 2)
        ps = (1 - mu_sigma) * ps + mu_sigma / sigma * (np.linalg.solve(L, x_new - x))
        chi_n = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        # Update covariance
        if np.linalg.norm(ps) / np.sqrt(1 - (1 - 2 / N) ** (self._iters / self._ds)) < (1.4 + 2 / (N + 1)) * chi_n:
            C = (1 - 2 * self._c1 * self._cmu) * C + self._c1 * (np.outer(ps, ps) + (1 - hsig) * C) \
                + self._cmu * sum(weights[i] * np.outer(Y[i] - x, Y[i] - x) for i in range(mu))
            hsig = 1.0
        else:
            C = (1 - 2 * self._c1 * self._cmu) * C + self._c1 * (np.outer(ps, ps) + C) \
                + self._cmu * sum(weights[i] * np.outer(Y[i] - x, Y[i] - x) for i in range(mu))
            hsig = 0.0
        # Ensure C is symmetric positive definite
        C = (C + C.T) / 2
        # Symmetrize and add small regularization
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        C = (eigvecs * eigvals) @ eigvecs.T
        C = (C + C.T) / 2
        # Update pc
        c_sigma = (4 + mu_sigma / N) / (N + 4 + 2 * mu_sigma / N)
        pc = (1 - c_sigma) * pc + c_sigma * (np.linalg.norm(ps) / chi_n > 1.4 + 2 / (N + 1)) * (x_new - x) / sigma
        # Update sigma
        sigma = sigma * np.exp(self._cs / self._ds * (np.linalg.norm(ps) / chi_n - 1))
        # Clip sigma
        sigma = np.clip(sigma, 1e-10, 100.0)
        # Check convergence
        converged = sigma < 1e-8 or np.max(eigvals) / np.min(eigvals) > 1e14
        return x_new, sigma, C, pc, ps, best_x, best_f, converged

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # 9 for n=5
        mu = pop_size // 2  # top half

        # Weights
        w = np.log(pop_size + 0.5) - np.log(np.arange(1, pop_size + 1))
        w = w / np.sum(w)
        w = w[:mu]
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # CMA-ES parameters
        c1 = (2 + 2 / (n + 1)) / ((n + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, (2 + (mu_eff / 2 - 2) / (n + 2)) / ((n + 2) ** 2 + mu_eff))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = n + mu_eff / 2 - 2
        c_sigma = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)

        self._c1 = c1
        self._cmu = cmu
        self._cs = cs
        self._ds = d_sigma
        self._iters = 0

        rng = np.random.default_rng(42)

        best_x_global = None
        best_f_global = np.inf

        # Multiple restarts to handle multimodality
        max_restarts = 5
        calls_used = 0

        for restart in range(max_restarts):
            # Initialize
            x = rng.uniform(-4, 4, n)
            fx = func(x)
            calls_used += 1
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()

            sigma = 1.0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            iters_in_restart = 0
            max_iters = max(1, self.budget - calls_used - (max_restarts - 1 - restart) * 20)

            for _ in range(max_iters):
                self._iters = iters_in_restart
                if time.time() > deadline:
                    break
                x, sigma, C, pc, ps, bx, bf, done = self._cma_es_step(
                    func, x, sigma, C, pc, ps, w, mu, n, pop_size, rng, deadline)
                iters_in_restart += 1
                calls_used += pop_size
                if bf < best_f_global:
                    best_f_global = bf
                    best_x_global = bx.copy()
                if done:
                    break
                if time.time() > deadline:
                    break

            if time.time() > deadline:
                break

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
