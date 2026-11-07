"""CMA-ES with multi-start for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES based optimizer for BBOB f19 (Griewank-Rosenbrock).
    Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) which adapts
    its search distribution to the local landscape geometry, making it well-suited
    for rotated/shifted smooth functions like Griewank-Rosenbrock.
    Multi-start with stratified sampling for global exploration, then CMA-ES
    for local exploitation.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                if not np.isfinite(f):
                    f = 1e10
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        def cma_es(x0, sigma0, max_evals=100):
            """
            CMA-ES implementation for local optimization.
            Adapts covariance matrix to the local landscape geometry.
            """
            nonlocal calls_made, best_x, best_f
            n = dim
            x = np.array(x0, dtype=np.float64)
            sigma = sigma0

            # CMA-ES parameters
            lam = max(4, 4 + int(3 * np.log(n)))  # population size
            mu = lam // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # State variables
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            chi_n = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))

            def update_C_sqrt_inv(C):
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    return True
                except:
                    return False

            update_C_sqrt_inv(C)

            # Initial population
            pop_x = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            for i in range(lam):
                z = np.random.randn(n)
                pop_x[i] = x + sigma * C_sqrt_inv @ z
                f_val = safe_call(pop_x[i])
                if f_val is None:
                    return x, np.inf
                pop_f[i] = f_val

            evals_used = lam
            no_improve_count = 0
            best_local_f = np.min(pop_f)

            while evals_used < max_evals and calls_made < budget and time.time() < deadline:
                # Select best mu individuals
                idx = np.argsort(pop_f)[:mu]
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (pop_x[idx[i]] - x)
                x_new = x + y_w

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (y_w / sigma)
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals_used / lam))) < (1.4 + 1/n) * chi_n:
                    c_sigma_eff = c_sigma
                else:
                    c_sigma_eff = 0

                # Update sigma
                sigma *= np.exp(c_sigma_eff / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = min(sigma, 50.0)
                sigma = max(sigma, 1e-8)

                # Update p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma

                # Update C
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (pop_x[idx[i]] - x) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Regularize C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update inverse sqrt
                if not update_C_sqrt_inv(C):
                    break

                x = x_new

                # Generate new population
                for i in range(lam):
                    z = np.random.randn(n)
                    pop_x[i] = x + sigma * C_sqrt_inv @ z
                    f_val = safe_call(pop_x[i])
                    if f_val is None:
                        return x, np.inf
                    pop_f[i] = f_val

                evals_used += lam

                # Check for improvement
                current_min = np.min(pop_f)
                if current_min < best_local_f - 1e-10:
                    best_local_f = current_min
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 5 and sigma < 1e-4:
                        break

            # Find best in final population
            best_idx = np.argmin(pop_f)
            return pop_x[best_idx], pop_f[best_idx]

        # --- Phase 1: Stratified multi-start sampling ---
        rng = np.random.RandomState(42)
        n_starts = 15
        start_points = []

        # Include origin
        start_points.append(np.zeros(dim))

        # Latin Hypercube sampling for good coverage
        for i in range(n_starts - 1):
            pt = np.zeros(dim)
            for d in range(dim):
                pt[d] = -4.0 + 8.0 * ((i + 1 + d) / (n_starts + 1))
            # Add small random jitter
            pt += rng.uniform(-0.5, 0.5, size=dim)
            start_points.append(pt)

        # Evaluate initial points
        for pt in start_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = safe_call(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            f = safe_call(best_x)
            if f is None:
                best_f = 1e10

        # --- Phase 2: CMA-ES from multiple starting points ---
        # Sort start points by their function values and run CMA-ES on the best ones
        start_results = []
        for pt in start_points:
            if calls_made >= budget or time.time() > deadline:
                break
            # Already evaluated, use cached best_f from safe_call
            # We need to track individual f values, so re-evaluate top candidates
            pass

        # Run CMA-ES from the global best first
        if calls_made < budget and time.time() < deadline - 30:
            remaining = budget - calls_made
            max_cma_evals = min(remaining // 2, 150)
            if max_cma_evals >= 20:
                x_cma, f_cma = cma_es(best_x, sigma0=1.0, max_evals=max_cma_evals)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()

        # Run CMA-ES from other promising starts
        if calls_made < budget and time.time() < deadline - 10:
            # Try CMA-ES from a few other starting points
            other_starts = [pt for pt in start_points if not np.allclose(pt, best_x)]
            rng2 = np.random.RandomState(123)
            rng2.shuffle(other_starts)

            for pt in other_starts[:3]:
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                remaining = budget - calls_made
                if remaining < 30:
                    break
                max_cma_evals = min(remaining // 3, 80)
                if max_cma_evals >= 15:
                    x_cma, f_cma = cma_es(pt, sigma0=0.8, max_evals=max_cma_evals)
                    if f_cma < best_f:
                        best_f = f_cma
                        best_x = x_cma.copy()

        # --- Phase 3: Final polish with CMA-ES on the best point ---
        if calls_made < budget and time.time() < deadline - 5:
            remaining = budget - calls_made
            if remaining >= 20:
                max_cma_evals = min(remaining, 60)
                x_final, f_final = cma_es(best_x, sigma0=0.3, max_evals=max_cma_evals)
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            f = safe_call(best_x)
            if f is None:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
