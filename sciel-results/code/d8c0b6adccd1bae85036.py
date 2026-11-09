"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES tuned for 5D BBOB f7 with coordinate descent refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters for dim=5
        lam = 4 + int(3 * np.log(dim))  # = 9 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        delta_heav = (1 - (1 - c_c) ** 2) / c_c if c_c > 0 else 0

        def time_ok():
            return (time.time() - start_time) < deadline

        def inv_sqrt_spd(C_mat):
            """Compute inverse square root of symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = inv_sqrt_spd(C)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # First evaluation
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # CMA-ES main loop
        while calls < budget and time_ok():
            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(ys) < mu:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sel = np.array([ys[i] for i in order[:mu]])

            # Update mean
            x_new = sum(weights[i] * ys_sel[i] for i in range(mu))

            # Update step-size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ (x_new - x_mean))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                sigma *= np.exp(c_sigma / 5.0 * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + delta_heav * C)
            for i in range(mu):
                y_diff = (ys_sel[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_diff, y_diff)

            C_inv_sqrt = inv_sqrt_spd(C)
            x_mean = x_new

        # Coordinate descent refinement
        if time_ok() and calls < budget:
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(20):  # 20 iterations of coordinate descent
                if not time_ok() or calls >= budget:
                    break
                improved = False
                for coord in range(dim):
                    if not time_ok() or calls >= budget:
                        break
                    for direction in [1, -1]:
                        if not time_ok() or calls >= budget:
                            break
                        x_try = x.copy()
                        x_try[coord] += direction * step
                        if np.any(x_try < -5) or np.any(x_try > 5):
                            continue
                        f_try = func(x_try)
                        calls += 1
                        if f_try < f:
                            x, f = x_try, f_try
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
