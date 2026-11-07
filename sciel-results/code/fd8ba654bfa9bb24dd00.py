import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        def count_calls(n=1):
            nonlocal total_calls
            total_calls += n

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5.0, 5.0)
            fy = float(func(x))
            total_calls += 1
            if fy < best_f:
                best_f = fy
                best_x = x.copy()
            return fy

        # Phase 1: Random exploration to find a good starting region
        # Use 30% of budget for random sampling
        n_random = budget // 3
        for _ in range(n_random):
            if not time_ok() or total_calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            evaluate(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES from the best point found
        # Use remaining budget for CMA-ES
        remaining = budget - total_calls
        if remaining <= 0 or not time_ok():
            return best_x, best_f

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        x = best_x.copy()
        sigma = 1.0  # Start with moderate step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        calls_in_cma = 0
        stagnation_count = 0
        f_prev = best_f

        # Precompute eigendecomposition cache
        eig_cache = None
        eig_cache_time = 0.0

        while total_calls < budget and time_ok():
            # Generate lam candidates
            candidates = []
            for i in range(lam):
                if not time_ok() or total_calls >= budget:
                    break
                z = rng.standard_normal(dim)
                
                # Use cached eigendecomposition if available and fresh
                now = time.time()
                if eig_cache is not None and (now - eig_cache_time) < 0.001:
                    evecs, evals_sqrt = eig_cache
                else:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.sqrt(np.maximum(evals, 1e-12))
                    eig_cache = (evecs, evals)
                    eig_cache_time = now
                
                y = x + sigma * (evecs @ (evals_sqrt * z))
                fy = evaluate(y)
                candidates.append((fy, y))
                calls_in_cma += 1

            if len(candidates) < mu:
                break

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])
            y_best = np.array([c[1] for c in candidates[:mu]])
            m = np.sum(weights[:, None] * y_best, axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m - x)

            # Covariance matrix update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_in_cma + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            if hsig:
                delta_h = 0
                pc = (1 - c_c) * pc + c_c * (m - x) / sigma
            else:
                delta_h = 1
                pc = (1 - c_c) * pc

            # Update covariance matrix
            C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * np.outer(pc, pc)
            # Ensure positive semi-definite
            evals_C, evecs_C = np.linalg.eigh(C)
            evals_C = np.maximum(evals_C, 1e-12)
            C = evecs_C @ np.diag(evals_C) @ evecs_C.T

            # Invalidate eig cache after C update
            eig_cache = None

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Update mean
            x = m

            # Stagnation check
            if candidates[0][0] >= f_prev - 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
            f_prev = candidates[0][0]

            # Restart if stagnated
            if stagnation_count > 15:
                # Perturb from best found
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                calls_in_cma = 0
                stagnation_count = 0
                f_prev = best_f
                eig_cache = None

        return best_x, best_f
