"""Improved BBOB f22 optimizer: Adaptive CMA-ES with multi-start and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive CMA-ES with multi-start exploration and Nelder-Mead polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def nelder_mead(x_start, f_start, max_steps=30, initial_delta=0.3):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
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
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                    break

        def cma_es(x0, sigma0, max_iter=100):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            lam = max(4, int(4 + 3 * np.log(dim)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            sqrtC = np.eye(dim)

            best_local_x = mean.copy()
            best_local_f = safe_func(mean)

            for it in range(max_iter):
                if not check_time() or calls_made >= self.budget:
                    break

                # Generate population
                pop = np.zeros((lam, dim))
                pop_f = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (sqrtC @ z)
                    x = np.clip(x, -5, 5)
                    pop[i] = x
                    pop_f[i] = safe_func(x)
                    if not check_time() or calls_made >= self.budget:
                        return

                # Sort
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                if pop_f[0] < best_local_f:
                    best_local_f = pop_f[0]
                    best_local_x = pop[0].copy()

                # Update mean
                new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi * (sqrtC @ (new_mean - mean) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chi
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma

                # Update covariance
                delta = (2 * hsig - 1) * cc * (2 - cc)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (pop[i] - mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                if np.min(eigenvalues) < 1e-20:
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

                # Update sigma
                sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update sqrt(C)
                sqrtC = eigenvectors @ np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T

                mean = new_mean
                mean = np.clip(mean, -5, 5)

                # Convergence check
                if np.linalg.norm(new_mean - mean) < 1e-12 or sigma < 1e-10:
                    break

                # Periodic re-centering if stuck
                if it % 20 == 19 and pop_f[0] - pop_f[-1] < 1e-8:
                    break

            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()
            return best_local_x, best_local_f

        # Phase 1: LHS Exploration
        n_lhs = 30
        points = []
        if calls_made + n_lhs <= self.budget:
            lhs_points = np.zeros((n_lhs, n))
            for i in range(n):
                perm = rng.permutation(n_lhs)
                u = (perm + 0.5) / n_lhs
                lhs_points[:, i] = -5 + 10 * u
            for x in lhs_points:
                if not check_time() or calls_made >= self.budget:
                    break
                fx = safe_func(x)
                points.append((fx, x.copy()))

        points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES on top candidates
        top_k = 4
        remaining = self.budget - calls_made
        cma_budget_per_start = max(20, remaining // (top_k + 1))

        for i in range(min(top_k, len(points))):
            if not check_time() or calls_made >= self.budget:
                break
            x_start = points[i][1]
            f_start = points[i][0]
            # Use adaptive initial sigma based on landscape
            sigma0 = 0.8 if i == 0 else 0.5
            cma_es(x_start, sigma0, max_iter=cma_budget_per_start)

        # Phase 3: Nelder-Mead polish on best point
        if check_time() and calls_made < self.budget and best_f < np.inf:
            nelder_mead(best_x.copy(), best_f, max_steps=40, initial_delta=0.2)

        # Phase 4: Final fine-tuning if budget remains
        if check_time() and calls_made < self.budget and best_f < np.inf:
            nelder_mead(best_x.copy(), best_f, max_steps=20, initial_delta=0.05)

        return best_x, best_f

# EVOLVE-BLOCK-END
