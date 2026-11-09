"""Black-box continuous optimizer on BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with global sampling and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Global phase: sample points across the domain to find good basins
        # Use stratified sampling for better coverage
        n_global = min(50, budget // 10)
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_global):
            # Stratified: divide each dim into 4 bins, pick one per stratum
            x = np.zeros(dim)
            for d in range(dim):
                bin_start = (i // (4 ** d)) % 4
                x[d] = -5 + (bin_start + rng.random()) * 2.5
            x = np.clip(x, -5, 5)
            fx = func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Local phase: CMA-ES refinement on top candidates
        # Use remaining budget for local optimization
        remaining = budget - n_global
        # Allocate to top 3 candidates
        n_top = min(3, len(candidates))
        alloc_per = remaining // n_top

        for idx in range(n_top):
            if time.time() > deadline:
                break
            fx, x = candidates[idx]
            # Run CMA-ES starting from this point
            x, fx = self._cmaes(func, x, fx, alloc_per, dim, rng, deadline)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f

    def _cmaes(self, func, x, fx, budget, dim, rng, deadline):
        """CMA-ES with adaptive sigma."""
        # Initialize CMA-ES parameters
        sigma = 1.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        cs = 1 / (sqrt(dim + 1) + 3)
        ds = 1 + 2 * max(0, (sqrt((dim + 1 - 3) * (dim + 3) - 2) - 2) / (dim + 4))
        cmu = 2 / ((dim + 4) ** 2 + 2)
        cc = (4 + dim / 3) / (dim + 4 + 2 / 3)

        lambda_ = 4 + int(np.floor(3 * np.log(dim)))  # population size
        # Weights: rank-based
        weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()

        # Precompute transformation matrix B and D
        D = np.sqrt(np.diag(C))
        B = np.linalg.inv(np.sqrtm(C)) if dim <= 10 else np.eye(dim)

        for it in range(budget):
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_new = np.clip(x + sigma * y, -5, 5)
                offspring.append((x_new, z, y))

            # Evaluate offspring
            for i in range(lambda_):
                offspring[i] = (offspring[i][0], offspring[i][1], offspring[i][2], func(offspring[i][0]))

            # Sort by fitness
            offspring.sort(key=lambda t: t[3])

            # Update best
            if offspring[0][3] < fx:
                fx = offspring[0][3]
                x = offspring[0][0].copy()

            # Update evolution paths
            y_w = np.zeros(dim)
            for i in range(int(mu_eff)):
                y_w += weights[i] * offspring[i][2]

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_w / sigma
            hsig = min(1, np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1)) / np.sqrt(mu_eff) * (1 + cc))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_w / sigma

            # Update covariance matrix
            C = (1 + cmu * (hsig - 1) * cc * (2 - cc)) * C
            for i in range(int(mu_eff)):
                y_i = offspring[i][2] / sigma
                C += cmu * weights[i] * np.outer(y_i, y_i)
            C -= cmu * (hsig - 1) * cc * (2 - cc) * np.outer(pc, pc)

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))

            # Update B and D periodically (expensive for large dim, but dim=5 is small)
            if it % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs

        return x, fx
# EVOLVE-BLOCK-END
