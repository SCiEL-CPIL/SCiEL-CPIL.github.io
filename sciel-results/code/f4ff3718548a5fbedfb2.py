# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with adaptive covariance for f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.max_calls = self.budget - 2
        self._deadline = None

    def __call__(self, func):
        start = time.time()
        self._deadline = start + 280.0
        d = self.dim
        rng = np.random.default_rng(42)

        self._calls = 0
        self._best_x = None
        self._best_f = np.inf

        def call(x):
            if self._calls >= self.max_calls:
                return None
            self._calls += 1
            return func(np.asarray(x, dtype=float))

        def time_ok():
            return time.time() < self._deadline

        # ---------------- Phase A: Quick Sobol initialization to find a good starting point ----------------
        n_init = 20
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_pts = sampler.random(n_init)
        except Exception:
            sobol_pts = rng.random((n_init, d))

        sobol_pts = sobol_pts * 10.0 - 5.0
        sobol_pts = np.clip(sobol_pts, -5.0, 5.0)

        best_init_x = None
        best_init_f = np.inf
        for pt in sobol_pts:
            if not time_ok() or self._calls >= self.max_calls:
                break
            fx = call(pt)
            if fx is None:
                break
            if fx < best_init_f:
                best_init_f, best_init_x = fx, pt.copy()
            if fx < self._best_f:
                self._best_f, self._best_x = fx, pt.copy()

        # ---------------- Phase B: CMA-ES from the best initialization point ----------------
        if best_init_x is not None and self._calls < self.max_calls and time_ok():
            self._cma_es(func, call, best_init_x.copy(), time_ok, rng)

        return self._best_x, self._best_f

    def _cma_es(self, func, call, x0, time_ok, rng):
        """CMA-ES with population size 4 for 5D problem."""
        d = self.dim
        pop_size = 4
        lam = pop_size
        mu = lam // 2  # number of parents

        # CMA-ES parameters
        sigma = 2.0  # initial step size
        mean = x0.copy()

        # Covariance matrix C (initially identity)
        C = np.eye(d)
        C_inv = np.eye(d)

        # Evolution path variables
        ps = np.zeros(d)  # step-size control path
        pc = np.zeros(d)  # covariance adaptation path

        # Update rates
        cs = (mu + 2) / (d + mu + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1 / mu) / ((d + 2) ** 2 + mu))

        # Damping factors
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (d + 1) - 1) - 1) + cs
        dampc = 1 + 2 * max(0, np.sqrt((mu - 1) / (d + 1) - 1)) - cs

        # Eigen-decomposition cache
        eig_vals = np.ones(d)
        eig_vecs = np.eye(d)
        eig_update_count = 0

        # Bounded sampling: reflect points that go out of bounds
        def sample_point(sigma_local):
            # Standard normal sample
            z = rng.standard_normal(d)
            # Apply CMA-ES transformation: x = mean + sigma * C^{1/2} z
            # Use Cholesky-like decomposition via eigen decomposition
            # For efficiency, we'll use a simpler approach with regularized C
            x_new = mean + sigma_local * (eig_vecs @ (np.sqrt(np.maximum(eig_vals, 1e-10)) * (eig_vecs.T @ z)))
            # Clip to bounds
            x_new = np.clip(x_new, -5.0, 5.0)
            return x_new

        # Main loop
        max_iterations = 100
        stagnation_count = 0

        for iteration in range(max_iterations):
            if not time_ok() or self._calls >= self.max_calls:
                break

            # Check if sigma is too small
            if sigma < 1e-8:
                break

            # Generate population
            population = []
            for i in range(lam):
                if self._calls >= self.max_calls or not time_ok():
                    break
                x_i = sample_point(sigma)
                f_i = call(x_i)
                if f_i is None:
                    break
                population.append((f_i, x_i))

            if len(population) < lam:
                break

            # Sort by fitness
            population.sort(key=lambda x: x[0])

            # Update best
            if population[0][0] < self._best_f:
                self._best_f, self._best_x = population[0][0], population[0][1].copy()

            # Check for stagnation
            if population[0][0] >= self._best_f - 1e-10:
                stagnation_count += 1
                if stagnation_count > 20:
                    break
            else:
                stagnation_count = 0

            # Select parents
            parents = [x for _, x in population[:mu]]

            # Compute weighted mean
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            w_max = weights[0]

            # New mean
            new_mean = np.sum([w * x for w, x in zip(weights, parents)], axis=0)

            # Update evolution paths
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv @ (new_mean - mean)) / sigma
            mu_eff = np.sum(weights) ** 2

            # pc update
            pc = (1 - dampc) * pc + dampc * cs / (1 - cs) * (new_mean - mean) / sigma

            # Update covariance matrix
            delta_hs = (np.sum(ps * ps) / (1 - (1 - cs) ** (2 * (iteration + 1))) - 1) * 2 / (1 - cs)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_hs - 1) * C)
            C += cmu * np.sum([w * np.outer(x - mean, x - mean) for w, x in zip(weights, parents)], axis=0)

            # Ensure positive semi-definite
            C = 0.5 * (C + C.T)
            C += 1e-10 * np.eye(d)

            # Update eigen decomposition (every 10 iterations for efficiency)
            eig_update_count += 1
            if eig_update_count % 10 == 0 or iteration == 0:
                try:
                    eig_vals, eig_vecs = np.linalg.eigh(C)
                    # Ensure positive eigenvalues
                    eig_vals = np.maximum(eig_vals, 1e-10)
                except Exception:
                    eig_vals = np.ones(d)
                    eig_vecs = np.eye(d)

            # Update sigma (step-size adaptation)
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update mean
            mean = new_mean

            # Ensure mean is within bounds
            mean = np.clip(mean, -5.0, 5.0)

        # Final refinement with (1+1)-ES if budget remains
        if self._best_x is not None and self._calls < self.max_calls and time_ok():
            self._refine_es(func, call, self._best_x.copy(), time_ok, rng)

    def _refine_es(self, func, call, x, time_ok, rng):
        """Simple (1+1)-ES refinement."""
        d = self.dim
        sigma = 0.1
        max_iter = 50
        no_improve = 0

        for _ in range(max_iter):
            if not time_ok() or self._calls >= self.max_calls:
                break

            delta = rng.standard_normal(d) * sigma
            x_new = x + delta
            x_new = np.clip(x_new, -5.0, 5.0)

            f_new = call(x_new)
            if f_new is None:
                break

            if f_new < self._best_f - 1e-10:
                x = x_new
                self._best_f, self._best_x = f_new, x.copy()
                sigma = min(sigma * 1.1, 1.0)
                no_improve = 0
            else:
                no_improve += 1
                sigma *= 0.9
                if no_improve > 10:
                    break

# EVOLVE-BLOCK-END
