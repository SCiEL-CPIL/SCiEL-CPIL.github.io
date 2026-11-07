"""CMA-ES based optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is a state-of-the-art derivative-free optimizer that adapts the
    covariance matrix of a multivariate normal distribution to the local
    landscape structure. This is particularly effective for rotated/shifted
    landscapes like BBOB functions where axis-aligned methods struggle.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 300s to be safe
        deadline = start_time + 300
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Initial standard deviation
        # For BBOB, the search space is [-5, 5]^n. A sigma of 1.0 or 2.0 is typical.
        # Let's start with 2.0 to cover the space initially.
        sigma = 2.0
        
        # Population size
        # Standard formula: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8.
        # Let's use a slightly larger population for better exploration early on, 
        # but not too large to waste budget.
        # lambda = max(4, int(4 + 3 * np.log(n)))
        # For n=5, this is 8.
        # Let's try 12 or 16 to be more robust? 
        # With 500 budget, 12 evals per generation means ~40 generations.
        # 8 evals per generation means ~60 generations.
        # CMA-ES usually needs O(n^2) to O(n^3) evals to converge.
        # n=5 is small. 500 evals is plenty.
        # Let's stick with the standard formula but maybe bump it up slightly.
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        # Let's use 12 for n=5 to balance exploration/exploitation.
        # Actually, standard CMA-ES performs well with default lambda.
        # Let's use lambda_ = 12 for n=5 to be safe.
        lambda_ = 12 
        
        # Initial mean
        # Start at the center of the box
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step sizes for parameters
        # Standard CMA-ES step sizes
        cmu = 1.0 / (4 + np.exp(n / 20))  # Covariance adaptation rate
        ccs = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)  # Path length adaptation rate
        dsigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1) + ccs
        csigma = (1 + ccs) / (n + dsigma)
        cc = (2 + 2 * np.sqrt((lambda_ / 2 - 1) / (n + 1))) / (n + 2)
        
        # Eigenvalue damping
        damps = 1.0
        # Actually, we can simplify the CMA-ES implementation by using a simplified version
        # or relying on the standard update rules.
        
        # For robustness and simplicity in this constrained environment, 
        # I will implement a simplified but effective CMA-ES variant 
        # or use a well-tested simplified version.
        
        # Let's use a simplified CMA-ES that tracks the mean and a single step size,
        # but adapts the covariance matrix.
        
        # State variables
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Path variables for cumulative step size adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute constants
        # Eigenvalue update frequency
        # We will update C every generation for simplicity, though standard CMA-ES 
        # might delay it. With n=5, it's cheap.
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)
        
        # Evaluate function wrapper
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = clip(x)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial evaluation of mean
        f_mean = evaluate(mean)
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            # Sample from N(mean, sigma^2 * C)
            # We need to sample from a multivariate normal with covariance sigma^2 * C.
            # We can do this by computing the Cholesky decomposition of C.
            # However, C is updated every generation.
            
            # To avoid expensive Cholesky every generation if n is small, it's fine.
            # n=5 is very small.
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add small diagonal
                C = np.eye(n)
                L = np.eye(n)
            
            # Sample lambda_ points
            # z ~ N(0, I)
            Z = np.random.randn(lambda_, n)
            # X = mean + sigma * (Z @ L.T)
            # Note: (Z @ L.T) gives samples from N(0, L @ L.T) = N(0, C)
            # So X = mean + sigma * Z @ L.T
            X = mean + sigma * (Z @ L.T)
            
            # Clip population to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.array([evaluate(x) for x in X])
            
            # Check if we ran out of budget
            if calls >= budget:
                break
            
            # Sort by fitness
            # Handle NaNs/Infs by putting them at the end
            # np.argsort handles this if we replace inf with a large number?
            # Actually, if evaluate returns inf, we should treat it as worst.
            # Let's sort indices.
            # If any F is inf, it will be sorted to the end if we use default sort?
            # np.argsort sorts inf to the end for ascending order.
            
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Top mu individuals
            # mu = lambda_ / 2
            mu = max(1, int(lambda_ / 2))
            
            # Weights
            # Standard weights: w_i = log(mu + 0.5) - log(i + 1) for i=0..mu-1
            # Normalized so sum(w_i) = 1
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            
            # mu_eff = 1 / sum(w_i^2)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Update mean
            # m_new = sum(w_i * x_i)
            # Use only finite values? If the best is inf, we have a problem.
            # If all are inf, break.
            if np.isinf(F_sorted[0]):
                break
            
            # Filter out non-finite values from the top mu?
            # If some top mu are inf, it's bad. But usually, the best ones are finite.
            # Let's assume the top mu are finite.
            X_top = X_sorted[:mu]
            # Check if any are inf
            if np.any(np.isinf(X_top)):
                # This shouldn't happen if F_sorted[0] is finite
                pass
            
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Update path p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * L^-1 * (mean_new - mean) / sigma
            # L^-1 is the inverse of Cholesky. We can solve L * y = v instead of inverting.
            # v = (mean_new - mean) / sigma
            v = (mean_new - mean) / sigma
            # Solve L * y = v
            try:
                y = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y = v # Fallback
            
            p_sigma = (1 - csigma) * p_sigma + csigma * np.sqrt(mu_eff) * y
            
            # Check step size adaptation
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| ~ sqrt(n) * (1 - 1/(4n) + ...)
            # Approx: c_sigma * (||p_sigma|| / c_sigma - 1) / d_sigma ? No.
            # Standard formula:
            # sigma = sigma * exp( c_sigma/d_sigma * ( ||p_sigma||/E||N(0,I)|| - 1 ) )
            # E||N(0,I)|| = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)
            # For n=5, this is approx sqrt(5) * 0.95... 
            # Let's use the approximation: sqrt(n) * (1 - 1/(4*n))
            E_norm = np.sqrt(n) * (1 - 1.0/(4.0*n))
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(csigma / dsigma * (norm_p_sigma / E_norm - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update path p_c
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) * C^-1/2 * (mean_new - mean) / sigma
            # C^-1/2 * v = L^-T * L^-1 * v ? No.
            # C = L L^T. C^-1/2 = L^-T L^-1? No.
            # If C = L L^T, then C^-1/2 is not simply L^-1.
            # However, we can compute C^-1/2 * v by solving C u = v and then... no.
            # Actually, we can use the eigenvalue decomposition or just solve.
            # But for p_c update, we need C^-1/2 * (mean_new - mean)/sigma.
            # Let's compute C_inv_half_v.
            # We can compute C^-1 and then sqrt? No, too slow.
            # Alternative: Use the fact that we have L.
            # C^-1/2 = L^-T * (L^-1) ? No.
            # Let's just use a simplified update or skip p_c if it's too complex?
            # No, p_c is important for rotation adaptation.
            
            # Let's compute C_inv_v = C^-1 * v
            # And then C_inv_half_v = C^-1/2 * v
            # There is a trick: C^-1/2 * v = L^-T * (L^-1 * v) ? No.
            # Let's use the eigenvalue decomposition of C for this step?
            # Or, we can update C first, then compute C^-1/2?
            # The standard algorithm updates C, then uses the new C for the next iteration's sampling.
            # But p_c update uses the OLD C.
            
            # Let's compute C_old_inv_half_v.
            # We can diagonalize C: C = Q diag(lambda) Q^T
            # C^-1/2 = Q diag(1/sqrt(lambda)) Q^T
            # This is O(n^3). n=5 is fine.
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_inv_half_v = C_inv_half @ v
            except np.linalg.LinAlgError:
                C_inv_half_v = v
            
            p_c = (1 - cc) * p_c + cc * np.sqrt(mu_eff) * C_inv_half_v
            
            # Update Covariance Matrix C
            # C = (1 - c_mu) * C + c_mu * sum(w_i * (y_i * y_i^T))
            # where y_i = L^-1 * (x_i - mean) / sigma
            # Note: We use the OLD L and OLD mean for y_i calculation?
            # Yes, y_i are the standardized residuals from the previous distribution.
            
            # Compute Y matrix
            # Y = (X_top - mean) / sigma
            # Y_std = L^-1 * Y
            # We need to solve L * Y_std^T = Y^T
            Y = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y.T).T
            except np.linalg.LinAlgError:
                Y_std = Y
            
            # C_new = (1 - c_mu) * C + c_mu * sum(w_i * Y_std[i] * Y_std[i]^T)
            # c_mu = c_mu * (1 - c_sigma/d_sigma * (||p_sigma||^2 - 1) ) ? 
            # Standard: c_mu = c_mu * (1 - c_sigma/d_sigma * (||p_sigma||^2 - 1) )
            # But often simplified to c_mu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n+2)^2 + mu_eff)
            # And then adjusted.
            # Let's use the standard c_mu formula:
            c_mu_base = 2.0 * (mu_eff - 2.0 + 1.0/mu_eff) / ((n + 2.0)**2 + mu_eff)
            c_mu = c_mu_base * (1.0 - csigma / dsigma * (norm_p_sigma**2 - 1.0))
            
            # Clamp c_mu to [0, 2/mu_eff]
            c_mu = np.clip(c_mu, 0.0, 2.0 / mu_eff)
            
            # Update C
            # Sum of outer products
            # Y_std is (mu, n). w is (mu,).
            # Weighted sum: sum(w_i * outer(Y_std[i], Y_std[i]))
            # Can be done as: Y_std.T @ (w[:, np.newaxis] * Y_std)
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            
            C = (1 - c_mu) * C + c_mu * outer_sum
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            
            # Update mean
            mean = mean_new
            
            # Check for convergence or stagnation
            # If sigma is very small, we might be stuck.
            # If best_f hasn't improved for a long time, we could restart?
            # For now, CMA-ES should handle it.
            
            # If the step size is extremely small, we might break
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
