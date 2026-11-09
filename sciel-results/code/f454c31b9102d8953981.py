"""
CMA-ES optimizer for BBOB f7 (Step ellipsoidal) in 5D.

Strategy: Standard CMA-ES with careful parameter tuning and robust handling
of the step-ellipsoidal function's non-smoothness.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES is specifically designed to handle rotated and scaled elliptical
    landscapes like the BBOB f7 (Step Ellipsoidal) function. It adapts both
    the step size (sigma) and the search distribution's covariance matrix (C)
    to align with the principal axes of the function, allowing it to navigate
    the 'steps' and 'ellipsoidal' features efficiently.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # 1. Initialization
        # Start near the center, as the minimum is guaranteed to be in [-4, 4]^5
        mean = np.zeros(dim)
        sigma = 1.5  # Initial step size, covering a significant portion of the box
        
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # Standard CMA-ES lambda
        mu = lambda_ // 2  # Number of best individuals used for recombination
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effectiveness of the recombination
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)  # Path length adaptation for CM
        cs = (mu_eff + 2) / (dim + mu_eff + 5)  # Path length adaptation for sigma
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)  # Covariance learning rate
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))  # Covariance learning rate for rank-mu update
        
        # Damping factor for sigma
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # State variables
        ps = np.zeros(dim)  # Path for sigma
        pc = np.zeros(dim)  # Path for covariance
        C = np.eye(dim)     # Covariance matrix
        inv_sqrt_C = np.eye(dim)  # Inverse square root of C (approx)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))  # Expected norm of N(0,I)
        
        # Initial evaluation
        f_mean = func(mean)
        calls_made = 1
        best_x = mean.copy()
        best_f = f_mean
        
        # Main loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Store old mean
            old_mean = mean.copy()
            
            # 2. Sample population
            # Update inverse square root of C periodically to save time, but for 5D it's cheap
            # We will recompute the Cholesky factorization every generation for robustness in 5D
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset it
                C = np.eye(dim)
                L = np.eye(dim)
            
            pop = np.empty((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = mean + sigma * L @ z
            
            # Clip to box [-5, 5]
            pop = np.clip(pop, -5, 5)
            
            # 3. Evaluate population
            f_vals = np.empty(lambda_)
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                f_vals[i] = func(pop[i])
                calls_made += 1
                
                # Update global best
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = pop[i].copy()
            
            if calls_made >= budget:
                break
            
            # 4. Sort by fitness
            idx = np.argsort(f_vals)
            pop_sorted = pop[idx]
            f_sorted = f_vals[idx]
            
            # 5. Update mean (recombination)
            mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # 6. Update paths
            # Calculate the step
            step = (mean - old_mean) / sigma
            
            # Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * inv_sqrt_C @ step
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * step
            
            # 7. Update sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # 8. Update covariance matrix C
            # Rank-1 update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * chi_n)
            hsig = 1 if hsig else 0
            
            delta_sig = (hsig - 1) * cc * (2 - cc)
            
            C = (1 + c1 * delta_sig - c1) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            y = np.array([inv_sqrt_C @ ((pop_sorted[i] - old_mean) / sigma) for i in range(mu)])
            C = C + cmu * np.sum(weights[:, None, None] * np.array([np.outer(y[i], y[i]) for i in range(mu)]), axis=0)
            
            # 9. Update inverse square root of C
            # For 5D, we can compute the eigen-decomposition or Cholesky
            # We need inv_sqrt_C such that inv_sqrt_C @ C @ inv_sqrt_C.T = I
            # This is the inverse of the square root.
            # Let's compute the eigenvalues and vectors
            try:
                evals, evecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                evals = np.maximum(evals, 1e-12)
                inv_sqrt_evals = 1.0 / np.sqrt(evals)
                inv_sqrt_C = evecs @ np.diag(inv_sqrt_evals) @ evecs.T
            except np.linalg.LinAlgError:
                inv_sqrt_C = np.eye(dim)
                C = np.eye(dim)
            
            # 10. Stop criteria
            # If sigma is too small or progress is minimal
            if sigma < 1e-12:
                break
            
            # Optional: If the best value hasn't improved for a long time, we could restart, 
            # but CMA-ES usually handles this. We'll rely on the budget.

        return best_x, best_f
# EVOLVE-BLOCK-END
