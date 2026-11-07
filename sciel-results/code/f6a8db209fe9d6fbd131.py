"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES Optimizer for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        n_global = 20
        for _ in range(n_global):
            if not check_time():
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)

        # Phase 2: CMA-ES
        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(d))  # Population size
        mu = lambda_ // 2  # Number of weights
        # Compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        
        # Initial mean
        m = best_x.copy()
        
        # Covariance matrix
        C = np.eye(d)
        # Eigendecomposition for sampling
        D = np.sqrt(np.diag(C))
        B = np.eye(d)
        
        # Cumulative step size control
        cs = 1.0
        hs = 0.0
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 2) / (d + 1))
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Termination conditions
        max_iterations = 50
        min_sigma = 1e-10
        max_sigma = 10.0
        
        for iteration in range(max_iterations):
            if not check_time():
                break
            
            # Update eigendecomposition
            try:
                B, D = np.linalg.eigh(C)
                D = np.sqrt(D)
            except np.linalg.LinAlgError:
                break
            
            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = D @ (B @ z)
                x_new = m + sigma * y
                x_new = np.clip(x_new, -5, 5)
                population.append(x_new)
            
            # Evaluate population
            f_values = []
            for x_new in population:
                fx = evaluate(x_new)
                if fx is None:
                    break
                f_values.append(fx)
            
            if len(f_values) < lambda_:
                break
            
            # Sort by function value
            order = np.argsort(f_values)
            population = [population[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(np.array(weights)[:, np.newaxis] * np.array(population[:mu]), axis=0)
            
            # Cumulative step size adaptation
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (D @ (B.T @ ((m - m_old) / (sigma * np.linalg.norm(m - m_old) + 1e-16)))) if np.linalg.norm(m - m_old) > 1e-16 else (1 - c_sigma) * p_sigma
            # Simplified: use direction of mean change
            if np.linalg.norm(m - m_old) > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (D @ (B.T @ ((m - m_old) / np.linalg.norm(m - m_old))))
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            hs = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1))) - (1 - c_sigma)) / np.sqrt(d) if iteration > 0 else 0
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1))) - 1))
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # Update covariance matrix
            delta_s = (1 + c_sigma * hs * (hs - 1)) * (m - m_old)
            
            # Rank-one update
            C = (1 + c_c * c_sigma * hs * (hs - 1) - c_1 - c_mu) * C
            
            # Rank-mu update
            for i in range(mu):
                y_i = (population[i] - m_old) / sigma
                C += c_mu * np.outer(y_i, y_i)
            
            # Ensure positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(d)
            
            # Check convergence
            if sigma < min_sigma * 10:
                break
            if np.linalg.norm(m - m_old) < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
