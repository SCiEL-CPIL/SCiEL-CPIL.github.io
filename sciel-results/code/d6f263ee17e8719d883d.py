"""Combined CMA-ES with coordinate-wise golden-section polish for Bent Cigar (f12)."""
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with coordinate-wise golden-section polish for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 275.0
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_cov_pos = (2 + 2 * mu_eff / n) / (n + 3 + 2 * mu_eff / n)
        c_cov_neg = (2 + mu_eff / n) / (n + 3 + 2 * mu_eff / n)

        # Global best tracking
        best_x = np.zeros(n)
        best_fx = func(best_x)
        calls_used = 1

        def cma_es_phase(x0, sigma0, max_calls, deadline):
            """Run CMA-ES from x0 with initial sigma0, using at most max_calls evaluations."""
            nonlocal best_x, best_fx, calls_used

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Compute C^{-1/2}
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            no_improve_count = 0
            stagnation_threshold = 12
            prev_best = best_fx

            for gen in range(1, 2000):
                if time.time() > deadline:
                    break
                if calls_used + lam > max_calls:
                    rem = max_calls - calls_used
                    if rem < 1:
                        break
                    for _ in range(rem):
                        if time.time() > deadline:
                            break
                        z = rng.standard_normal(n)
                        y = x_mean + sigma * (C_inv_sqrt @ z)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < best_fx:
                            best_fx = fy
                            best_x = y.copy()
                    break

                # Sample population
                ys = np.zeros((lam, n))
                fs = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    ys[i] = x_mean + sigma * (C_inv_sqrt @ z)
                    ys[i] = np.clip(ys[i], -5, 5)
                    fs[i] = func(ys[i])
                    calls_used += 1
                    if fs[i] < best_fx:
                        best_fx = fs[i]
                        best_x = ys[i].copy()

                # Check improvement
                if best_fx < prev_best - 1e-10:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                prev_best = best_fx

                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = ys[idx]

                # Update mean
                x_mean_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)

                # Update evolution paths
                y_vec = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                # Update covariance
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                rank_mu = np.outer(pc, pc) + delta_h * np.outer(pc, pc)
                rank_1 = np.outer(y_vec, y_vec)
                C = (1 + c_sigma * (c_cov_pos - c_sigma) + c_cov_neg * (mu_eff - 2) - c_sigma * c_cov_neg) * C \
                    + c_cov_pos * rank_1 + c_cov_neg * (np.sum(w) - 1) * rank_mu

                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 100.0)

                # Update C^{-1/2}
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

                x_mean = x_mean_new

                # Stagnation handling: restart from best with larger sigma
                if no_improve_count >= stagnation_threshold and sigma > 0.05:
                    x_mean = best_x.copy()
                    sigma = max(sigma * 2.0, 0.5)
                    sigma = np.clip(sigma, 0.1, 50.0)
                    C = np.eye(n)
                    D, B = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-12)
                    C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    no_improve_count = 0

            return best_x, best_fx, calls_used

        def coordinate_polish(x0, max_calls, deadline):
            """Coordinate-wise golden-section search polish from x0."""
            nonlocal best_x, best_fx, calls_used

            if max_calls < 2:
                return

            x = x0.copy()
            golden_ratio = (1 + np.sqrt(5)) / 2
            inv_phi = 1 / golden_ratio

            # Number of iterations per dimension
            # Each iteration uses 2 function evaluations
            iters_per_dim = max(1, max_calls // (2 * n))

            for iteration in range(iters_per_dim):
                if time.time() > deadline or calls_used >= max_calls:
                    break

                for d in range(n):
                    if time.time() > deadline or calls_used >= max_calls:
                        break

                    # Current value at x[d]
                    x_curr = x[d]
                    f_curr = best_fx if np.allclose(x, best_x) else None

                    # Define interval around current best
                    # Use a range that adapts based on how far we are
                    span = 2.0
                    a = max(-5, x_curr - span)
                    b = min(5, x_curr + span)

                    if b - a < 1e-6:
                        continue

                    # Golden section search
                    c = b - inv_phi * (b - a)
                    d_point = a + inv_phi * (b - a)

                    # Evaluate at c and d
                    x_c = x.copy()
                    x_c[d] = c
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < best_fx:
                        best_fx = f_c
                        best_x = x_c.copy()

                    if time.time() > deadline or calls_used >= max_calls:
                        break

                    x_d = x.copy()
                    x_d[d] = d_point
                    f_d = func(x_d)
                    calls_used += 1
                    if f_d < best_fx:
                        best_fx = f_d
                        best_x = x_d.copy()

                    # Narrow the interval
                    if f_c < f_d:
                        b = d_point
                        d_point = c
                        f_d = f_c
                        c = b - inv_phi * (b - a)
                    else:
                        a = c
                        c = d_point
                        f_c = f_d
                        d_point = a + inv_phi * (b - a)

                    # Update x[d] to the best point found
                    if f_c <= f_d:
                        x[d] = c
                    else:
                        x[d] = d_point

                    # If we found a new best, update x to best_x
                    if np.allclose(x, best_x):
                        pass
                    else:
                        # Keep x as the current search point
                        pass

            # Final update: ensure best_x is the global best
            return best_x, best_fx, calls_used

        # Phase 1: CMA-ES with main budget
        cma_budget = int(budget * 0.7)
        cma_es_phase(np.zeros(n), 2.0, cma_budget, deadline)

        # Phase 2: Coordinate-wise golden-section polish with remaining budget
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline - 5:
            coordinate_polish(best_x.copy(), remaining, deadline)

        return best_x, best_fx
