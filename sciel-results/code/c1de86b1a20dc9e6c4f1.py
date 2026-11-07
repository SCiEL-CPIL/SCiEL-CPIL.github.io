"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f14.
    
    f14 (Different powers) has an anisotropic landscape where each dimension
    is raised to a different power. CMA-ES adapts a full covariance matrix to
    align the search distribution with the principal axes, making it highly
    effective for such landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls_made = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Diversified initialization
        n_init = min(10, budget)
        candidates = [np.zeros(dim)]
        for i in range(dim):
            for val in [-3.0, 3.0]:
                c = np.zeros(dim)
                c[i] = val
                candidates.append(c)
        while len(candidates) < n_init:
            candidates.append(rng.uniform(-4.0, 4.0, dim))

        for c in candidates[:n_init]:
            if calls_made >= budget or time.time() > deadline:
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: CMA-ES
        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 3 * mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_sigma = c_sigma * (1 + d_sigma)

        x = best_x.copy()
        f_best = best_f
        sigma = 1.0

        # Covariance matrix and its Cholesky decomposition
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute
        chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))

        # Main CMA-ES loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for sampling from N(0, C)
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                population.append((fi, xi))
                if fi < f_best:
                    f_best = fi
                    best_x = xi.copy()

                if calls_made >= budget or time.time() > deadline:
                    break

            if len(population) < lambda_:
                break

            # Sort by fitness
            population.sort(key=lambda p: p[0])
            selected = [p[1] for p in population[:mu]]

            # Update mean
            x_new = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)

            # Update path and covariance
            x_old = x
            x = x_new

            # Compute step
            step = (x - x_old) / sigma if sigma > 1e-15 else np.zeros(dim)

            # Update pc
            pc = (1 - c_c) * pc + c_c * step

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x - x_old)

            # Update covariance
            delta_heavy = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) * (c_sigma - 1)
            C = ((1 + c_sigma * (2 - c_sigma) - c_c) * C
                 + c_sigma * (pc[:, None] * pc[None, :]
                              - delta_heavy * C)
                 + c_sigma * (ps[:, None] * ps[None, :] / (sigma ** 2) * mu_eff
                              - np.eye(dim) * mu_eff))

            # Ensure C is symmetric positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, f_best
# EVOLVE-BLOCK-END
