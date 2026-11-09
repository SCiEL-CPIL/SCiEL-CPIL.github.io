"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Initial population
        n = max(4 + int(3 * np.log(dim)), dim + 2)
        n = min(n, budget // 5)  # ensure we have enough budget

        # Start from origin (center of box)
        x_mean = np.zeros(dim)
        sigma = 3.0

        # Generate initial population
        xs = []
        fs = []
        for i in range(n):
            xi = x_mean + sigma * rng.standard_normal(dim)
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            xs.append(xi)
            fs.append(fi)

        # Sort by fitness
        order = np.argsort(fs)
        xs = [xs[i] for i in order]
        fs = [fs[i] for i in order]

        # CMA-ES parameters
        lam = n
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State variables
        ps = np.zeros(dim)
        hs = np.zeros(dim)
        C = np.eye(dim)
        step_size = sigma
        pc = np.zeros(dim)

        # Track best
        best_x = xs[0].copy()
        best_f = fs[0]
        eval_count = lam

        # Precompute weights for selection
        w_sel = weights[:mu]

        # Main loop
        while eval_count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            xs = []
            fs = []
            for i in range(lam):
                # Sample from multivariate normal with covariance C
                z = rng.standard_normal(dim)
                xi = x_mean + step_size * (C @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                xs.append(xi)
                fs.append(fi)
                eval_count += 1

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()

            # Weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w_sel[i] * xs_sorted[i]

            # Update CMA-ES state
            diff = x_new - x_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / step_size * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (dim + 1)) * chi
            pc = (1 - cc) * pc + hsig * cc / step_size * diff

            # Rank-mu update
            yk = [xs_sorted[i] - x_mean for i in range(mu)]
            C_new = C
            for i in range(mu):
                C_new += w_sel[i] * np.outer(yk[i], yk[i])

            # Covariance adaptation
            C = (1 + c1 * (hsig - 1) * np.outer(pc, pc) + cmu * (C_new - C)) * (1 - c1)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = (C + C.T) / 2
                C += 1e-10 * np.eye(dim)

            x_mean = x_new

            # Adapt step size
            step_size *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            step_size = np.clip(step_size, 1e-10, 10.0)

            # Check for convergence
            if np.all(np.abs(C - np.eye(dim) * np.trace(C) / dim) < 1e-6):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
