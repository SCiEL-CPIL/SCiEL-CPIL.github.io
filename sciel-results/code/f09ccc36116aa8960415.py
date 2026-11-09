"""Improved BBOB f22 optimizer: Adaptive CMA-ES with multi-restart and NM polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 270.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling to find promising basins
        n_random = min(80, max(20, self.budget // 6))
        candidates = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))

        candidates.sort(key=lambda t: t[0])

        # CMA-ES implementation
        def cma_es(x0, f0, max_evals, sigma0=0.5):
            nonlocal best_x, best_f, calls_made

            if max_evals <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            pop_size = max(4, int(4 + 3 * np.log(dim)))

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Adaptation rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # State
            x_mean = np.array(x0, dtype=float).copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            evals_used = 0
            no_improve = 0
            best_local_f = f0

            while evals_used < max_evals and check_time() and (self.budget - calls_made) > 0:
                # Generate population
                samples = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    # Apply covariance
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    y = L @ z
                    x_new = np.clip(x_mean + sigma * y, -5, 5)
                    samples.append(x_new)

                # Evaluate
                fvals = []
                for x_new in samples:
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    f_new = safe_func(x_new)
                    fvals.append(f_new)
                    evals_used += 1

                if len(fvals) < pop_size:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                samples = [samples[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check improvement
                if fvals[0] < best_local_f:
                    best_local_f = fvals[0]
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                x_mean_new = sum(w * samples[i] for i, w in enumerate(weights))
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update step-size control (CSA)
                y_mean = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                pc = (1 - c_c) * pc + c_c * y_mean
                delta_heav = (1 + (1 - c_c) * c_c) * (np.linalg.norm(pc) ** 2 / dim - 1)
                C = (1 + c_1 + c_mu - c_c) * C + c_1 * (pc @ pc.T - C)
                outer = np.zeros((dim, dim))
                for i, w in enumerate(weights):
                    outer += w * np.outer(samples[i] - x_mean, samples[i] - x_mean)
                C += c_mu * outer / sigma ** 2

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x_mean = x_mean_new

                # Stop if sigma is very small or no improvement for many iterations
                if sigma < 1e-7 or no_improve > 3 * pop_size:
                    break

            return

        # Nelder-Mead polish
        def nelder_mead(x0, f0, max_steps=60):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            shrink = 0.5

            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                delta = 0.1 if abs(x0[i]) < 1 else 0.05
                xi[i] += delta * (1 if xi[i] < 0 else -1)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, dim + 1):
                if not check_time() or (self.budget - calls_made) <= 0:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fxr = safe_func(xr)
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                if fxr < simplex_f[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or (self.budget - calls_made) <= 0:
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
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + shrink * (simplex[i] - simplex[0]), -5, 5)
                            if not check_time() or (self.budget - calls_made) <= 0:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                    break

        # Phase 2: CMA-ES from top candidates
        top_k = min(3, len(candidates))
        for i in range(top_k):
            if not check_time() or (self.budget - calls_made) <= 10:
                break
            remaining = self.budget - calls_made
            max_evals = min(remaining - 10, 150)
            if max_evals > 10:
                x_start = candidates[i][1].copy()
                f_start = candidates[i][0]
                cma_es(x_start, f_start, max_evals, sigma0=0.5)

        # Phase 3: Nelder-Mead polish on best
        if best_x is not None and check_time() and (self.budget - calls_made) > 15:
            nelder_mead(best_x, best_f, max_steps=60)

        # Phase 4: Final CMA-ES refinement if budget allows
        if best_x is not None and check_time() and (self.budget - calls_made) > 20:
            remaining = self.budget - calls_made
            max_evals = min(remaining - 5, 80)
            if max_evals > 10:
                cma_es(best_x, best_f, max_evals, sigma0=0.2)

        # Phase 5: Final NM polish
        if best_x is not None and check_time() and (self.budget - calls_made) > 10:
            nelder_mead(best_x, best_f, max_steps=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
