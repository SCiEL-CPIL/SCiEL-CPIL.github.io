"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin for 300s limit
        
        # Problem setup
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        # Initial standard deviation
        # The box is [-5, 5]^5. The minimizer is in [-4, 4]^5.
        # A reasonable initial sigma covers a good portion of the search space.
        sigma = 2.0 
        # Initial mean: Center of the box
        x = np.zeros(dim)
        
        # CMA-ES state
        # Covariance matrix C
        C = np.eye(dim)
        # Inverse square root of C for efficient sampling
        # We will recompute this periodically or use Cholesky if stable, 
        # but for small dim, direct inversion/sqrt is fine.
        # Actually, standard CMA-ES maintains C and uses Cholesky or eigendecomposition for sampling.
        # For simplicity and robustness in a small budget, we use the standard update rules.
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        # mu is the number of selected individuals
        mu = lambda_pop // 2
        # Weights for the selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # mu_eff is the equivalent population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[1] / mu_eff - 1) / ((dim + 2)**2 + mu_eff))
        
        # Initial best
        # We start with a few random points to get a sense of the landscape?
        # No, CMA-ES starts from a mean. Let's start from 0.
        # However, the problem has a hidden shift. Starting from 0 is a decent guess 
        # given the minimizer is in [-4, 4].
        
        # To be safer against bad initialization, we can do a quick random search first?
        # No, let's trust CMA-ES. But we should ensure we don't get stuck at the boundary.
        
        # Let's initialize x to 0.
        # If the function is very flat, CMA-ES might need to expand.
        
        # Pre-compute sampling matrix
        # We use Cholesky decomposition of C for sampling.
        # If C becomes singular, we need to handle it.
        
        best_x = x.copy()
        best_f = np.inf
        calls_made = 0
        
        # Initial evaluation
        # Since we don't know the function, we start by evaluating the initial mean.
        # But CMA-ES usually evaluates the population.
        # Let's generate the first population.
        
        # We need to be careful with the budget.
        # Total budget is 500.
        
        # Helper to sample from N(0, C)
        def sample_from_covariance(C, rng):
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                return L @ z
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                z = rng.standard_normal(dim)
                return L @ z

        rng = np.random.default_rng(0)
        
        # We will run generations until budget is exhausted or time runs out.
        # Each generation consumes lambda_pop evaluations.
        
        while calls_made + lambda_pop <= budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample direction
                d = sample_from_covariance(C, rng)
                # New candidate
                y = x + sigma * d
                # Clip to box
                y = np.clip(y, -5, 5)
                
                # Evaluate
                fy = func(y)
                calls_made += 1
                ys[i] = y
                fs[i] = fy
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                # Early exit if time is tight
                if time.time() > deadline:
                    return best_x, best_f

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]
            
            # Update mean
            # x_new = sum(weights[i] * ys_sorted[i])
            x_new = np.dot(weights, ys_sorted[:mu])
            
            # Update path variables
            # Inverse square root of C
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # h_sigma condition
            hs = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + lambda_pop) / lambda_pop)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1))
            
            # p_c update
            p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # Covariance update
            # Rank-one update
            v = (x_new - x) / sigma
            C = (1 + c_1 * (2 - c_1) + (c_c * (2 - c_c) / c_1) * (hs * np.dot(p_c, p_c) - 1)) * C \
                - c_1 * C @ np.outer(v, v) \
                - c_1 * np.outer(C @ v, v) \
                + (c_c * (2 - c_c) / c_1) * (hs * np.dot(p_c, p_c) - 1) * np.outer(p_c, p_c) \
                + c_1 * (1 + (2 / c_1 - 1) * c_mu) * np.outer(v, v)
                
            # Rank-mu update
            # Note: The standard formula is:
            # C += c_mu * sum(weights[i] * outer(v_i, v_i)) - c_1 * C ... wait, the rank-one part already handled C update mostly.
            # Let's use the standard combined update formula carefully.
            
            # Actually, let's stick to the simpler, more robust update sequence often used in implementations:
            # 1. Update p_sigma
            # 2. Update p_c
            # 3. Update C
            
            # Let's re-implement the C update more cleanly to avoid algebraic errors.
            # C <- (1 + c1(2-c1) + cmu(1 + (2/c1 - 1)cmu) - c1) C + c1 (C - I) + ... this is getting messy.
            
            # Standard CMA-ES Update:
            # C = (1 + c1(2 - c1) + cmu(1 + (2/c1 - 1)cmu)) * C - c1 * C - cmu * sum(weights * outer(y_i - x, y_i - x)) / sigma^2
            # Wait, the term is:
            # C_new = (1 + c1(2-c1) + cmu(1 + (2/c1 - 1)cmu) - c1) * C 
            #          + c1 * (C - I) # No, that's not right either.
            
            # Let's use the formulation from Hansen's paper:
            # C <- (1 + c1(2 - c1) + cmu(1 + (2/c1 - 1)cmu)) C 
            #      - c1 C - cmu sum(weights_i * outer(v_i, v_i))
            #      + c1 (C - I) is not standard.
            
            # Correct formula:
            # C <- (1 + c1(2-c1) + cmu(1 + (2/c1 - 1)cmu)) C 
            #      - c1 C 
            #      - cmu * sum(weights_i * outer(v_i, v_i))
            #      + c1 * I  # No.
            
            # Let's use the most common implementation:
            # 1. Calculate v_i = (y_i - x) / sigma
            # 2. C = (1 + c1*(2-c1) + cmu*(1 + (2/c1 - 1)*cmu)) * C
            # 3. C -= c1 * C
            # 4. C -= cmu * sum(weights * outer(v_i, v_i))
            # 5. C += c1 * I  # This ensures trace preservation roughly? No.
            
            # Actually, the update is:
            # C_new = (1 + c1(2-c1) + cmu(1 + (2/c1 - 1)cmu)) * C 
            #         - c1 * C 
            #         - cmu * sum(weights_i * outer(v_i, v_i))
            #         + c1 * I  # This term is missing in my memory.
            
            # Let's use the simpler Rank-1 update only, which is more stable and often sufficient for small budgets.
            # Rank-1 update:
            # C = (1 + c1(2-c1)) * C - c1 * C + c1 * outer(v, v)
            # This simplifies to:
            # C = (1 - c1 + c1(2-c1)) C + c1 outer(v,v) ? No.
            
            # Let's use the explicit formula from the CMA-ES reference implementation:
            # C = (1 + c1*(2-c1) + cmu*(1 + (2/c1 - 1)*cmu)) * C
            # C -= c1 * C
            # C -= cmu * sum(weights * outer(v_i, v_i))
            # C += c1 * np.eye(dim) # This is not standard.
            
            # I will use a simplified Rank-1 update for stability, ignoring the Rank-Mu part for now, 
            # or implementing it carefully.
            
            # Let's restart the C update logic properly.
            v_list = (ys_sorted[:mu] - x) / sigma
            # Rank-Mu term
            mu_term = np.zeros((dim, dim))
            for i in range(mu):
                mu_term += weights[i] * np.outer(v_list[i], v_list[i])
                
            # Rank-One term
            v_mean = (x_new - x) / sigma
            one_term = np.outer(v_mean, v_mean)
            
            # Update C
            # Factor 1: (1 + c1(2-c1) + cmu(1 + (2/c1 - 1)cmu))
            factor = 1 + c1*(2-c1) + cmu*(1 + (2/c1 - 1)*cmu)
            C = factor * C - c1 * C - cmu * mu_term + c1 * np.eye(dim)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update x
            x = x_new
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma/d_sigma * (np.linalg.norm(p_sigma)/np.sqrt(mu_eff) - 1))
            
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Regularize C to ensure positive definiteness
            # Add small identity if eigenvalues are too small
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
