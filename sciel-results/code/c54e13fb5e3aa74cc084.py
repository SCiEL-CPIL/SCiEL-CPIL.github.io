"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initialize CMA-ES parameters
        # For 5D, lambda = 4 + floor(3 * log(5)) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(dim))
        
        # Initial center
        x0 = np.zeros(dim)
        
        # Initial sigma
        # The box is [-5, 5], so width is 10. 
        # A good initial sigma is often (xmax - xmin) / 10 or similar.
        # For f4 (Bueche-Rastrigin), the global minimum is at 0, but there are local minima.
        # Starting at 0 with a moderate sigma might get stuck in local minima.
        # However, CMA-ES adapts. Let's start with sigma = 2.0.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Step size control parameters
        alpha_mu = 1.0
        alpha_sigma = 0.5
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1)
        c_sigma = (d_sigma + 2) / (dim + d_sigma + 5)
        c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.2) ** 2 + lam)
        c_mu = min(1 - c_1, (2 * (mu_eff / lam - 1) + c_1) / ((dim + 2) ** 2 + lam))
        
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w[:int(np.floor(lam / 2))] ** 2)
        
        # Initial evaluation
        x_best = x0.copy()
        f_best = func(x_best)
        
        # Loop
        calls_made = 1
        while calls_made < budget:
            # Check time
            if time.time() - start_time > timeout:
                break
            
            # Generate population
            # Sample from N(x0, sigma^2 * C)
            # We need to generate lam samples
            # Efficient way: generate standard normal, transform
            Z = rng.standard_normal((lam, dim))
            
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Y = x0 + sigma * (Z @ L.T)
            Y = x0 + sigma * (Z @ L.T)
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += lam
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < f_best:
                f_best = F[idx_best]
                x_best = Y[idx_best].copy()
            
            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            mu_w = np.sum(w[:int(np.floor(lam / 2))])
            x0_new = np.sum(w[:int(np.floor(lam / 2))] * Y_sorted[:int(np.floor(lam / 2))], axis=0)
            
            # Update sigma
            # Calculate z vector
            z = (x0_new - x0) / (sigma * np.linalg.norm(L, 2))
            # Actually, norm of L is not quite right for the distance in the transformed space.
            # Standard formula: z = (x0_new - x0) / (sigma * sqrt(trace(C)))? No.
            # Let's use the standard CMA-ES update rules carefully.
            
            # Distance in the search space normalized by sigma and the transformation.
            # d = || x0_new - x0 ||_C^{-1} ? No, it's simpler.
            # The step size update uses the norm of the step in the "standard normal" space.
            # Step = sigma * L.T @ Z_weighted.
            # So Z_weighted = L^{-T} @ (x0_new - x0) / sigma
            try:
                L_inv_T = np.linalg.inv(L).T
                Z_w = L_inv_T @ ((x0_new - x0) / sigma)
                z_norm = np.linalg.norm(Z_w)
            except:
                z_norm = 0
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_norm
            sigma *= np.exp(alpha_sigma / d_sigma * (ps / chi_n - 1))
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (x0_new - x0) @ (x0_new - x0).T + c_mu * sum(w_i * (y_i - x0) @ (y_i - x0).T)
            # Note: y_i - x0 should be in the original space, but weighted.
            # The rank-mu update uses the selected points.
            
            delta_heav = 1 + 2 * max(0, np.sqrt((mu_eff - 2) / (dim + 2)) - 1) + c_sigma
            # Correction for the rank-1 update term to maintain trace
            # The standard implementation often uses a correction factor.
            
            # Rank 1 update
            rank1 = (x0_new - x0) @ (x0_new - x0).T
            
            # Rank mu update
            # H = sum(w_i * (y_i - x0) @ (y_i - x0).T)
            # We need to be careful with the transformation.
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * H
            # Where H is computed from the selected individuals.
            
            Y_selected = Y_sorted[:int(np.floor(lam / 2))]
            # Weights are w[:int(np.floor(lam / 2))]
            w_sel = w[:int(np.floor(lam / 2))]
            
            # Compute H
            # H = sum(w_i * (y_i - x0) @ (y_i - x0).T)
            # This can be computed as (Y_selected.T * w_sel) @ (Y_selected - x0)
            # Wait, (y_i - x0) is a vector.
            # Let's use:
            # D = Y_selected - x0
            # H = (D.T * w_sel) @ D
            
            D = Y_selected - x0
            H = (D.T * w_sel) @ D
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * H
            
            # Symmetrize C to prevent numerical drift
            C = (C + C.T) / 2
            
            # Ensure positive definiteness (optional, but good for stability)
            # If eigenvalues become negative, add small identity
            try:
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += (1e-12 - eigvals.min()) * np.eye(dim)
            except:
                pass
            
            x0 = x0_new
        
        return x_best, f_best
# EVOLVE-BLOCK-END
