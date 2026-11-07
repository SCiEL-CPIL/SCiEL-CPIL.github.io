"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, budget, dim, x0, sigma0, deadline):
        n = dim
        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2
        # weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # step-size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        # covariance adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        # evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        C = np.eye(n)
        # initial values
        x = x0.copy()
        sigma = sigma0
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        # precompute
        chol = np.linalg.cholesky(C)
        start_time = time.time()
        
        while calls < budget:
            # check time
            if time.time() > deadline:
                break
            # generate population
            ys = []
            xs = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (chol @ z)
                y = np.clip(y, -5, 5)
                ys.append(z)
                xs.append(y)
            # evaluate
            fs = np.array([func(xi) for xi in xs])
            calls += lambda_
            # update best
            idx = np.argmin(fs)
            if fs[idx] < best_f:
                best_f = fs[idx]
                best_x = xs[idx].copy()
            
            # sort
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order[:mu]]
            xs_sorted = [xs[i] for i in order[:mu]]
            fs_sorted = fs[order]
            
            # new center of mass
            m = sum(weights[i] * xs_sorted[i] for i in range(mu))
            
            # update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (chol @ (m - x))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * (pc @ pc.T - C) + c_mu * sum(weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))
            # ensure symmetry and positive definite
            C = (C + C.T) / 2
            # recompute Cholesky
            try:
                chol = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                chol = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
            
            # update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)
            
            # update mean
            x = m
            x = np.clip(x, -5, 5)
            
            # check stagnation - if no improvement in last 2*lambda evaluations, try restart
            if calls > 4 * lambda_ and fs_sorted[0] >= best_f * 1.0001:
                # simple restart: perturb best point
                x = best_x + 0.5 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                sigma = 0.5
                C = np.eye(n)
                chol = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
        
        return best_x, best_f, calls

    def __call__(self, func):
        global rng
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin
        
        # Phase 1: Random restarts to find good initial point
        # Use 50 evaluations for random search
        n_random = min(50, budget // 10)
        best_x = None
        best_f = np.inf
        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES from best point
        x0 = best_x
        sigma0 = 1.0
        remaining = budget - n_random
        
        # Run CMA-ES
        best_x, best_f, calls = self._cma_es(func, remaining, dim, x0, sigma0, deadline)
        total_calls = n_random + calls
        
        # If we still have budget and time, do more random restarts
        while total_calls < budget and time.time() < deadline - 10:
            # Try a new random start
            x_new = rng.uniform(-5, 5, dim)
            fx_new = func(x_new)
            total_calls += 1
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
                # Run CMA-ES from this new point
                remaining = budget - total_calls
                if remaining > 20:
                    x0 = best_x
                    sigma0 = 0.5
                    bx, bf, calls = self._cma_es(func, remaining, dim, x0, sigma0, deadline)
                    total_calls += calls
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()
            # Check time
            if time.time() > deadline - 5:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
