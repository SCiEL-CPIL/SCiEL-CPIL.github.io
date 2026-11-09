"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive covariance matrix for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            # Handle potential non-finite values by clipping to valid range or default
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Initialize CMA-ES parameters
        # Start from a random point or origin? Origin is often safe, but random might be better for unknown shift.
        # The problem states minimizer is in [-4, 4]^5. Let's start at 0 or a random point.
        # A random start within the box is safer against bad initialization at 0 if 0 is far.
        # However, CMA-ES is robust. Let's use a random start in [-2, 2] to be close to center.
        m = rng.uniform(-2, 2, dim)
        
        # Evaluate initial mean to get a baseline
        f_m = call_func(m)
        best_x = m.copy()
        best_f = f_m

        # Population size lambda
        lambda_pop = 4 + int(3 * np.log(dim))
        # Ensure we don't exceed budget
        if lambda_pop > budget // 5:
            lambda_pop = max(2, budget // 10)

        # Weights for selection
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initial step size
        sigma = 1.0
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C for sampling
        # Use eigen decomposition for stability
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Counter for covariance update (every 1/mu_eff generations)
        count_cs = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Check convergence: sigma too small or spread too small
            spread = sigma * np.sqrt(np.trace(C))
            if spread < 1e-12 * (np.max(np.abs(m)) + 1.0):
                break

            # Sample population
            pop = []
            for _ in range(lambda_pop):
                # Sample from N(0, I), transform to N(0, C)
                z = rng.standard_normal(dim)
                x = m + sigma * (C_inv_sqrt @ z)
                pop.append(x)

            # Evaluate population
            f_vals = []
            for x in pop:
                fx = call_func(x)
                f_vals.append(fx)
                
                # Early stop if budget is hit
                if calls >= budget:
                    break

            if len(f_vals) < 2:
                break

            f_vals = np.array(f_vals)
            # Sort by fitness
            order = np.argsort(f_vals)
            pop_sorted = [pop[i] for i in order]
            f_sorted = f_vals[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()

            # Update mean
            m_old = m.copy()
            m = np.sum([weights[i] * pop_sorted[i] for i in range(mu)], axis=0)

            # Update sigma (CSA)
            # Calculate norm of p_sigma update
            z_m = C_inv_sqrt.T @ ((m - m_old) / sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_m
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Clamp sigma to avoid numerical issues
            sigma = np.clip(sigma, 1e-15, 1e5)

            # Update C
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt.T @ ((m - m_old) / sigma)
            
            # Heff correction term
            heff = np.sum(weights) ** 2
            delta_sigma = (1 - (1 - c_c)**2) * (np.linalg.norm(p_c)**2 / mu_eff - 1)
            
            C = (1 - c_1 - c_mu + c_1 * delta_sigma) * C
            C += c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            y = np.array([(pop_sorted[i] - m_old) / sigma for i in range(mu)])
            C += c_mu * np.sum([weights[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)

            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update C_inv_sqrt periodically or if C changes significantly
            # For simplicity and stability, update every generation if dim is small (5)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
