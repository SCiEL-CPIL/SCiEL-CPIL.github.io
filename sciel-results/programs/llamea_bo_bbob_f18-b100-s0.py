import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # State tracking
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        rng = np.random.default_rng(42)

        # 1. Initial Global Exploration
        # Spend a small fraction of the budget on random sampling to find a good starting point
        n_init = max(10, int(budget * 0.05))
        for _ in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)

        # 2. CMA-ES Optimization
        # Parameters tuned for dim=5, budget=500
        lam = 15
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        def run_cma_es(x0, sigma0, max_gens=None):
            nonlocal n_calls, best_x, best_f, budget
            if n_calls >= budget or time.time() > deadline:
                return
            
            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            stagnation_count = 0
            stagnation_threshold = 10
            max_restarts = 3
            restarts = 0

            while n_calls + lam <= budget and time.time() < deadline:
                if max_gens and (n_calls // lam) > max_gens:
                    break
                
                # Generate population
                Z = rng.standard_normal((lam, dim))
                X = m + sigma * (Z @ L.T)
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([timed_call(x) for x in X])
                
                if n_calls >= budget:
                    break

                # Sort
                order = np.argsort(F)
                F_sorted = F[order]
                X_sorted = X[order]

                # Stagnation check
                if F_sorted[0] >= best_f:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

                # If stagnated, restart
                if stagnation_count > stagnation_threshold:
                    if restarts >= max_restarts:
                        break
                    restarts += 1
                    # Restart from best_x with smaller sigma
                    m = best_x.copy()
                    sigma *= 0.5
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(dim)
                    stagnation_count = 0
                    continue

                # Update Mean
                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                
                # Update Sigma (CSA)
                try:
                    diff = (m_new - m) / sigma
                    y = np.linalg.solve(L.T, diff)
                    z = np.linalg.solve(L, y)
                except:
                    z = diff

                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z
                norm_ps = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / chi_lambda * (norm_ps / chi_lambda - 1))
                sigma = np.clip(sigma, 1e-12, 100.0)

                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * z

                # Update C
                C_rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_diff = (X_sorted[i] - m) / sigma
                    C_rank_mu += weights[i] * np.outer(y_diff, y_diff)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 + (1 - c_c) * (2 - c_c)) * c_c * (1 - c_c) * np.eye(dim)) + c_mu * C_rank_mu
                C = (C + C.T) / 2

                # Ensure positive definite
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(dim)
                        C = np.eye(dim)

                m = m_new

        # Run CMA-ES
        run_cma_es(best_x, 2.0)

        # 3. Final Polish
        # If budget remains, do a few local random steps or Nelder-Mead
        if n_calls < budget and time.time() < deadline:
            # Simple local refinement with adaptive step size
            step_size = 0.1
            remaining = budget - n_calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = np.clip(best_x + step_size * rng.standard_normal(dim), -5.0, 5.0)
                f = timed_call(x)
                if f < best_f:
                    step_size *= 0.9
                else:
                    step_size *= 1.1
                    step_size = min(step_size, 1.0)

        return best_x, best_f
