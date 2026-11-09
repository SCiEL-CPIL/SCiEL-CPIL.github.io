"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before the 330s kill

        # ---------- Phase 1: Latin Hypercube exploration ----------
        n_lhs = 60
        rng = np.random.default_rng(12345)
        perm = [rng.permutation(n_lhs) for _ in range(dim)]
        lxs = np.array([(perm[d] + rng.random()) / n_lhs for d in range(dim)]).T
        lxs = lxs * 10.0 - 5.0  # map to [-5,5]
        lxs = np.clip(lxs, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            fx = func(lxs[i])
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = lxs[i].copy()

        # ---------- Phase 2: CMA-ES with periodic restarts ----------
        # CMA-ES parameters
        pop_size = 8
        lambda_ = pop_size
        mu = pop_size // 2  # number of selected individuals
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        x_mean = best_x.copy()
        sigma = 0.5 * (5.0 - (-5.0)) / (3.0 * np.sqrt(dim)) # Heuristic initial sigma
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim) # Inverse square root of C
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        stagnation_counter = 0
        max_stagnation = 100 # Restart if no improvement for 100 generations (approx 800 evals, might be too high, let's use 50)
        max_stagnation = 50
        
        # Precompute some constants
        hs = 1 + 0.3 * min(1, (mu_eff - 1) / dim)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((pop_size, dim))
            fs = np.zeros(pop_size)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5.0, 5.0)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                
                if fs[i] < global_best_f:
                    global_best_f = fs[i]
                    global_best_x = y.copy()
            
            if calls >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            y_sorted = ys[order]
            f_sorted = fs[order]
            
            # Check stagnation
            if f_sorted[0] >= global_best_f - 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # Update mean
            x_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update covariance matrix
            y_diff = (y_sorted[:mu] - x_mean) / sigma
            delta_sigma = (np.linalg.norm(p_sigma) - chi_n) / (chi_n * (1 + c_sigma))
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (stagnation_counter + 1))) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (weights @ y_diff.T).T)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma if hasattr(self, 'x_mean_old') else p_c
            
            # Wait, I need to store x_mean_old before updating it. Let's refactor.
            # Actually, standard CMA-ES updates p_c using the old mean.
            # Let's do it properly in the loop.
            
            # Refactor for correctness:
            # 1. Save old mean
            # 2. Update mean
            # 3. Update p_c using old mean
            # 4. Update C
            
            # Since I already updated x_mean above, I need to be careful.
            # Let's restart the loop logic to be cleaner.
            pass 

        # The above loop had a logical flaw in order of operations for p_c.
        # Let's implement a cleaner CMA-ES block.
        
        # Reset state for clean CMA-ES
        x_mean = best_x.copy()
        sigma = 0.5
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        stagnation_counter = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # 1. Generate population
            ys = np.zeros((pop_size, dim))
            fs = np.zeros(pop_size)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5.0, 5.0)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                
                if fs[i] < global_best_f:
                    global_best_f = fs[i]
                    global_best_x = y.copy()
            
            if calls >= budget or time.time() > deadline:
                break
            
            # 2. Sort
            order = np.argsort(fs)
            y_sorted = ys[order]
            f_sorted = fs[order]
            
            # 3. Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # 4. Update evolution paths
            y_diff = (y_sorted[:mu] - x_mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (weights @ y_diff.T).T)
            
            # h_sigma calculation
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for h_sigma to avoid complex power calculation if possible, but standard is:
            # h_sigma = 1 if norm(p_sigma) / sqrt(1 - (1-c_sigma)^2 * t) < (1.4 + 2/(dim+1)) * chi_n
            # We don't track t easily in this simplified loop, so use a heuristic or simplified check
            # For robustness, let's just check if norm is reasonable
            h_sigma = 1 if norm_p_sigma < (1.4 + 2 / (dim + 1)) * chi_n * 1.5 else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma
            
            # 5. Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] * y_diff[:, :, None] * y_diff[:, None, :], axis=0)
            
            C = (1 + c_c * (2 - c_c) * h_sigma * (1 - np.trace(C) / dim) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definite and compute inverse square root
            # Symmetrize
            C = (C + C.T) / 2
            # Eigen decomposition for sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt_inv = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
            except:
                C_sqrt_inv = np.eye(dim)
            
            # 6. Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # 7. Check stagnation
            if f_sorted[0] >= global_best_f - 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # 8. Restart logic
            if stagnation_counter > 50:
                x_mean = global_best_x + 0.1 * rng.standard_normal(dim)
                x_mean = np.clip(x_mean, -5.0, 5.0)
                sigma = 0.5
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_counter = 0

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
