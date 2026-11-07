"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # CMA-ES parameters
        # For dim=5, a population size of 4-6 is often good. Let's use 4 for speed and efficiency with 500 budget.
        N = 4 
        mu = N // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (N + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
        c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        c_1 = 2 / ((N + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        
        chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        
        # Start near center, but slightly offset to avoid symmetry issues if any? 
        # Standard is 0.0.
        x_mean = np.zeros(self.dim)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix C and its decomposition B, D
        C = np.eye(self.dim)
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0
        
        def check_time():
            return time.time() > deadline

        # Initial evaluation
        # Ensure we don't exceed budget immediately (unlikely but safe)
        if calls < self.budget:
            fx = func(x_mean)
            calls += 1
            best_f = fx
            best_x = x_mean.copy()

        # Main loop
        while calls < self.budget and not check_time():
            # Generate population
            pop = np.empty((N, self.dim))
            f_vals = np.empty(N)
            
            # Generate all N points
            for i in range(N):
                if calls >= self.budget or check_time():
                    break
                z = rng.standard_normal(self.dim)
                # Transform z using B and D
                # x = mean + sigma * B @ (D * z)
                x_i = x_mean + sigma * (B @ (D * z))
                # Clip to box [-5, 5]
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i
                
                f_i = func(x_i)
                calls += 1
                f_vals[i] = f_i
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            # If we didn't complete the generation, break to avoid partial updates
            if calls >= self.budget or check_time():
                break
            
            # Sort population by fitness
            order = np.argsort(f_vals)
            # Select the best mu
            selected_indices = order[:mu]
            selected_pop = pop[selected_indices]
            
            # Weighted mean
            new_x_mean = np.sum(weights[:, None] * selected_pop, axis=0)
            
            # Update covariance matrix adaptation
            y = (new_x_mean - x_mean) / sigma
            pc = (1 - c_c) * pc + c_c * y
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (new_x_mean - x_mean)
            
            # Check for sigma adaptation (hsig)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (N + 1)) * chi_N
            
            if hsig:
                pc = (1 - c_c) * pc + c_c * y
                delta_heav = 0
            else:
                delta_heav = c_c * (2 - c_c)
            
            # Update C
            # C = (1 - c_1 - c_mu + delta_heav) * C + c_1 * (pc @ pc.T + (1 - hsig) * delta_heav * C) + c_mu * sum(weights_i * (x_i - mean) @ (x_i - mean) / sigma^2)
            
            # Calculate rank-mu update
            diff = selected_pop - x_mean
            # weights shape (mu,), diff shape (mu, dim)
            # We need sum_i w_i * (diff_i @ diff_i.T) / sigma^2
            # Using einsum or broadcasting
            # diff[:, None, :] * diff[:, None, :] -> (mu, 1, dim) * (mu, 1, dim) is wrong.
            # diff shape (mu, dim). outer(diff_i, diff_i) is (dim, dim).
            # We want sum_i w_i * outer(diff_i, diff_i)
            rank_mu_update = np.einsum('i,ik,il->kl', weights, diff, diff)
            
            C = (1 - c_1 - c_mu + delta_heav) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * delta_heav * C) + c_mu * rank_mu_update
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
            
            # Update mean
            x_mean = new_x_mean
            
            # Recompute B and D from C
            # Eigendecomposition
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))
            
            # Optional: restart if sigma is too small or mean is out of bounds?
            # For now, keep it simple.

        return best_x, best_f
# EVOLVE-BLOCK-END
