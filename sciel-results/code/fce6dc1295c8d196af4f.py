"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock) in 5D.
    
    Strategy:
    1. Small global random search (5 points) for early progress.
    2. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for robust,
       fast convergence on the Rosenbrock valley.
    3. Strict budget and time management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        dim = self.dim

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x_arr = np.asarray(x, dtype=float)
            # Ensure x is within bounds [ -5, 5 ]
            x_arr = np.clip(x_arr, -5.0, 5.0)
            fx = float(func(x_arr))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_arr.copy()
            return fx

        # Phase 1: Global Random Search
        # Use 5 points to quickly find a decent basin.
        # This ensures early AOCC gain.
        global_points = 5
        for _ in range(global_points):
            if not time_ok() or call_count >= self.budget:
                break
            x = rng.uniform(-5, 5, dim)
            tracked_func(x)

        # Phase 2: CMA-ES
        # CMA-ES is highly effective for Rosenbrock.
        # Parameters:
        #   lambda (population size)
        #   sigma (step size)
        #   C (covariance matrix)
        #   p_sigma, p_c (evolution paths)
        
        if time_ok() and call_count < self.budget - 10:
            # Initialize CMA-ES parameters
            # Population size
            lam = int(4 + 3 * np.log(dim))
            # Initial step size
            sigma = 1.0
            # Mean vector
            m = best_x.copy()
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Learning rates
            # Standard CMA-ES rates
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Pre-compute Cholesky factor
            C_sqrt = np.linalg.cholesky(C)
            
            # CMA-ES Loop
            while time_ok() and call_count < self.budget - 5:
                # Generate population
                y_list = []
                x_list = []
                f_list = []
                
                for i in range(lam):
                    if not time_ok() or call_count >= self.budget:
                        break
                    y = rng.standard_normal(dim)
                    # Transform y to x
                    x = m + sigma * (C_sqrt @ y)
                    x = np.clip(x, -5.0, 5.0)
                    f = tracked_func(x)
                    y_list.append(y)
                    x_list.append(x)
                    f_list.append(f)
                
                if len(f_list) < 2:
                    break

                # Sort by function value
                idx = np.argsort(f_list)
                y_list = [y_list[i] for i in idx]
                x_list = [x_list[i] for i in idx]
                f_list = [f_list[i] for i in idx]
                
                # Update mean
                m_new = np.sum([weights[i] * x_list[i] for i in range(mu)], axis=0)
                
                # Update sigma
                # Check for convergence of sigma
                if sigma < 1e-12:
                    break
                
                # Update paths and covariance
                # p_sigma update
                y_mean = np.sum([weights[i] * y_list[i] for i in range(mu)], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (y_mean / sigma))
                
                # h_sigma condition
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (call_count // lam)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
                
                # p_c update
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
                
                # Update C
                # delta_h for h_c
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (call_count // lam)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
                delta_h = (1 - h_c) * c_c * (2 - c_c)
                
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z = (x_list[i] - m) / sigma
                    rank_mu += weights[i] * np.outer(z, z)
                
                C = (1 + c_1 * (h_sigma - 1) + c_mu - c_1) * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2 # Symmetrize
                
                # Ensure positive definite
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If not PD, restart or add small identity
                    C += 1e-6 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        break

                m = m_new
                
                # Check convergence
                if np.max(f_list[:mu]) - np.min(f_list[:mu]) < 1e-14:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
