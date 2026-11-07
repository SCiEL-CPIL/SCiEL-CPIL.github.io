"""Restart-based CMA-ES with L-BFGS-B polish for BBOB f19 (Griewank-Rosenbrock).

Structural change: multiple independent short CMA-ES restarts from diverse
initial points, followed by a final L-BFGS-B polish. This handles the global
structure of the Griewank term better than a single long run, because the
landscape has multiple basins and a single CMA-ES trajectory can get trapped.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Restart-based CMA-ES optimizer with final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_short(self, x0, f0, rng, budget, deadline, sigma0=1.5):
        """Run CMA-ES for at most `budget` evaluations, respecting deadline.
        Returns (best_x, best_f, n_calls)."""
        n = self.dim
        if budget < 4:
            return x0, f0, 0

        pop = max(4, min(12, int(4 + 3 * np.log(n))))
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)

        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs

        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        c = np.array(x0, dtype=float).copy()
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x = c.copy()
        best_f = f0
        calls = 0
        it = 0
        last_best_f = f0
        no_improve = 0

        while it < budget and calls < budget and time.time() < deadline:
            Y = np.zeros((pop, n))
            F = np.empty(pop)

            for i in range(pop):
                if calls >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                x = np.clip(x, -5.0, 5.0)
                F[i] = float(self._f(x))
                Y[i] = y
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if calls >= budget or time.time() >= deadline:
                break

            # If not all evaluations completed, pad with worst
            if not np.all(np.isfinite(F)):
                break

            idx = np.argsort(F)
            Ys = Y[idx]
            Fs = F[idx]

            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean

            c = c + cs * d
            c = np.clip(c, -5.0, 5.0)

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean

            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
                hsig = 1.0
            else:
                hsig = 0.0

            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))

            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-5, 10.0))

            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                Csqrt = (evecs * np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                Csqrt = np.eye(n)
                sigma = sigma0 * 0.5
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                c = best_x.copy()

            it += 1

            if best_f == last_best_f:
                no_improve += pop
            else:
                last_best_f = best_f
                no_improve = 0

            if no_improve > 4 * pop:
                break

        return best_x, best_f, calls

    def _lbfgsb_polish(self, x0, f0, deadline):
        """Final L-BFGS-B polish. Returns (best_x, best_f, n_calls)."""
        calls = 0
        best_x = np.array(x0, dtype=float).copy()
        best_f = f0

        def fobj(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = float(self._f(x))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            res = minimize(fobj, x0, method='L-BFGS-B',
                           bounds=[(-5.0, 5.0)] * self.dim,
                           options={'maxiter': 200, 'maxfun': 150,
                                    'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            pass

        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(12345)
        self._f = func

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Multiple short CMA-ES restarts from diverse initial points
        # Use 5 restarts, each with ~120 budget, keeping the best
        n_restarts = 5
        budget_per_restart = int(0.75 * self.budget / n_restarts)

        # Generate diverse initial points
        init_points = []
        # Origin
        init_points.append(np.zeros(self.dim))
        # Random points in the box
        for _ in range(n_restarts - 1):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))

        for i, x_init in enumerate(init_points):
            if total_calls >= self.budget or time.time() >= deadline - 30:
                break
            if i == 0:
                f_init = float(func(x_init))
                total_calls += 1
            else:
                # Evaluate initial point
                f_init = float(func(x_init))
                total_calls += 1

            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()

            remaining = self.budget - total_calls
            alloc = min(budget_per_restart, remaining)
            if alloc < 10:
                break

            x_best, f_best, n_calls = self._cmaes_short(
                x_init, f_init, rng, alloc, deadline, sigma0=1.5)
            total_calls += n_calls
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        if best_x is None:
            x0 = np.zeros(self.dim)
            best_f = float(func(x0))
            total_calls += 1
            best_x = x0.copy()

        # Phase 2: L-BFGS-B polish if we have budget and time
        remaining = self.budget - total_calls
        if remaining >= 20 and time.time() < deadline - 15:
            x_pol, f_pol, n_pol = self._lbfgsb_polish(best_x, best_f, deadline)
            total_calls += n_pol
            if f_pol < best_f:
                best_f = f_pol
                best_x = x_pol.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
