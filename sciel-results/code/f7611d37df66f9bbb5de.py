"""BBOB f22 optimizer: CMA-ES with adaptive restarts for Gallagher 21 peaks."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f22 with multi-restart strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.timeout = 290.0

    def _eval(self, x, func):
        if self.calls >= self.budget:
            return np.inf
        if time.time() - self.start_time > self.timeout:
            return np.inf
        self.calls += 1
        x = np.clip(x, -5.0, 5.0)
        f = func(x)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _cma_es(self, func, x0, sigma0, budget_remaining, max_iter=None):
        """Run CMA-ES from initial point x0 with step size sigma0."""
        n = self.dim
        if max_iter is None:
            max_iter = budget_remaining

        # CMA-ES parameters
        pop_size = int(4 + 3 * np.log(n))
        pop_size = max(pop_size, 6)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_cov = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_cov_mu = (2 + 2 * mu_eff / n) / (n + 4 + mu_eff / n)

        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_cov = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        best_local_f = self.best_f
        best_local_x = self.best_x.copy()

        for iteration in range(max_iter):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.timeout:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            ys = rng.standard_normal((pop_size, n))
            xs = x_mean + sigma * (ys @ B.T)
            xs = np.clip(xs, -5.0, 5.0)

            # Evaluate
            fs = np.array([self._eval(x, func) for x in xs])

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]

            # Update best
            if fs_sorted[0] < best_local_f:
                best_local_f = fs_sorted[0]
                best_local_x = xs_sorted[0].copy()

            # Compute weighted mean
            y_w = (weights[:, None] * (xs_sorted[:mu] - x_mean)).sum(axis=0)

            # Update p_sigma
            h_sig = 1.0 if np.sqrt(1 - (1 + iteration/max(1, iteration+1))**(2*(1+c_sigma)/c_sigma)) / np.sqrt(1-c_sigma) < (1.4 + 2/(n+1)) * chi_n else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B_inv_sqrt @ y_w / sigma

            # Update mean
            x_mean = x_mean + c_sigma * y_w

            # Update covariance
            delta_h = (1 - h_sig) * c_sigma * (2 - c_sigma)
            p_cov = (1 - c_cov) * p_cov + delta_h * np.sqrt(c_cov * (2 - c_cov) * mu_eff) * (B_inv_sqrt @ y_w / sigma)
            C = (1 + c_cov * (c_cov_mu + delta_h) - c_cov) * C + c_cov_mu * (y_w @ y_w.T / sigma**2)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(n)))
            except:
                C_inv_sqrt = np.eye(n)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Convergence check
            if sigma < 1e-10:
                break

            # Stagnation check
            if iteration > 20 and np.std(fs_sorted[:mu]) < 1e-12:
                break

        return best_local_x, best_local_f

    def _nelder_mead(self, func, x_start, f_start, max_iter=50):
        """Nelder-Mead local optimization."""
        n = self.dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        shrink = 0.5

        simplex = [x_start.copy()]
        simplex_f = [f_start]
        for i in range(n):
            xi = x_start.copy()
            step = 0.1
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            if np.allclose(xi, x_start):
                xi[i] = x_start[i] - step
                xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                return x_start, f_start
            fx = self._eval(simplex[i], func)
            simplex_f.append(fx)

        best_nm_f = min(simplex_f)
        best_nm_x = simplex[np.argmin(simplex_f)].copy()

        for step in range(max_iter):
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if simplex_f[0] < best_nm_f:
                best_nm_f = simplex_f[0]
                best_nm_x = simplex[0].copy()

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = self._eval(xr, func)
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                break

            if fxr < simplex_f[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = self._eval(xe, func)
                if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                    break
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_f[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
            elif fxr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fxr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fxc = self._eval(xc, func)
                if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                    break
                if fxc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fxc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                            return best_nm_x, best_nm_f
                        simplex_f[i] = self._eval(simplex[i], func)

            if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                break

        return best_nm_x, best_nm_f

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        rng = np.random.default_rng(42)

        n = self.dim
        budget = self.budget

        # Phase 1: LHS initialization to find diverse starting points
        n_lhs = 20
        lhs_points = np.zeros((n_lhs, n))
        for i in range(n):
            perm = rng.permutation(n_lhs)
            u = (perm + 0.5) / n_lhs
            lhs_points[:, i] = -5 + 10 * u

        lhs_results = []
        for i in range(n_lhs):
            if self.calls >= budget or time.time() - self.start_time > self.timeout:
                break
            fx = self._eval(lhs_points[i], func)
            lhs_results.append((fx, lhs_points[i].copy()))

        if not lhs_results:
            return self.best_x, self.best_f

        lhs_results.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from top starting points
        # Use remaining budget for CMA-ES
        remaining = budget - self.calls
        if remaining > 30:
            # Pick top 3 starting points for multi-start CMA-ES
            n_starts = min(3, len(lhs_results))
            cma_budget_per_start = max(10, remaining // (n_starts + 1))

            for s in range(n_starts):
                if self.calls >= budget or time.time() - self.start_time > self.timeout:
                    break

                x0 = lhs_results[s][1]
                sigma0 = 2.0 if s == 0 else 1.0

                best_x_cma, best_f_cma = self._cma_es(
                    func, x0, sigma0, 
                    budget_remaining=cma_budget_per_start,
                    max_iter=min(cma_budget_per_start, 100)
                )

                if best_f_cma < self.best_f:
                    self.best_f = best_f_cma
                    self.best_x = best_x_cma.copy()

        # Phase 3: Final Nelder-Mead polish
        if self.calls < budget and time.time() - self.start_time < self.timeout and self.best_f < np.inf:
            nm_budget = budget - self.calls
            if nm_budget > 10:
                best_nm_x, best_nm_f = self._nelder_mead(
                    func, self.best_x, self.best_f,
                    max_iter=min(50, nm_budget)
                )
                if best_nm_f < self.best_f:
                    self.best_f = best_nm_f
                    self.best_x = best_nm_x.copy()

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
