"""CMA-ES with restarts for BBOB-style f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # standard CMA-ES pop size
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = 2 * weights[0] ** 2 / (1 - weights[0] ** 2)  # mu_eff related

        # Restart strategy: split budget across restarts
        # First restart gets more budget, subsequent ones get less
        n_restarts = 3
        budgets_per_restart = []
        remaining = budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                b = remaining
            else:
                b = max(50, int(remaining * 0.5))
            budgets_per_restart.append(b)
            remaining -= b
        # Ensure we don't exceed budget
        total = sum(budgets_per_restart)
        if total > budget:
            budgets_per_restart[-1] = budget - sum(budgets_per_restart[:-1])

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if remaining < 10:
                break

            b_r = budgets_per_restart[r]
            if b_r < 10:
                break

            # Initialize
            if r == 0:
                x = np.zeros(dim)
                sigma = 2.0
            else:
                x = rng.uniform(-4, 4, dim)
                sigma = 1.5

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES state
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)  # We'll use Cholesky for C
            Cs = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Step sizes
            c_sigma = (mu + 2) / (dim + mu + 5)
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            d_sigma = 1 + 2 * max(0, (np.sqrt((mu + 1) / (dim + 1)) - 1) / (dim + 3)) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            iter_count = 0
            for _ in range(b_r - 1):
                if time.time() > deadline:
                    break
                if iter_count >= b_r - 1:
                    break

                # Generate population
                # Use Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                Y = rng.standard_normal((pop_size, dim))
                # Transform: x + sigma * L @ Y^T  (row-wise)
                Z = np.einsum('ij,kj->ik', Y, L.T)
                P = x + sigma * Z
                P = np.clip(P, -5, 5)

                # Evaluate
                Fs = np.array([func(P[i]) for i in range(pop_size)])
                iter_count += pop_size

                # Track best
                idx_best = np.argmin(Fs)
                if Fs[idx_best] < best_f:
                    best_f = Fs[idx_best]
                    best_x = P[idx_best].copy()

                # Sort
                order = np.argsort(Fs)
                P_sorted = P[order]
                Fs_sorted = Fs[order]

                # Weighted mean
                m_new = np.sum(weights[:, None] * P_sorted[:mu], axis=0)

                # Update sigma (CSA)
                hs = np.linalg.norm(m_new - x) / (np.sqrt(2 * (dim + 1)) * sigma + 1e-12)
                hsig = (1 - hs / chi_n) / (1 + c_sigma) if hs < 1.5 * chi_n else 1.0

                # Update ps
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu) * np.linalg.inv(L) @ (m_new - x) / sigma

                # Update pc
                hpc = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + c_c) * chi_n)
                pc = (1 - c_c) * pc + hpc * np.sqrt(c_c * (2 - c_c) * mu) * (m_new - x) / sigma

                # Update C
                delta_h = (2 * hpc - 1) * c_c
                C = (1 + c_c * (2 - c_c) - c_sigma) * C \
                    + c_sigma * (1 / (1 - c_sigma) - 1 + delta_h) * np.outer(pc, pc) \
                    + c_sigma * (np.outer(ps, ps) - C) * (1 - hsig) / (1 - c_sigma)

                # Symmetrize and ensure positive definite
                C = 0.5 * (C + C.T)
                # Add small regularization if needed
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals.min() < 1e-8:
                        C += (1e-8 - eigvals.min()) * np.eye(dim)
                except:
                    pass

                # Update x and sigma
                x = m_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)

                # Stop if sigma is too small or C is degenerate
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
