"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        # Population size heuristic: 4 + floor(3 * log(n))
        # For n=5, lambda = 14. This is a good balance for exploration/exploitation.
        lam = max(4, 4 + int(3 * np.log(n)))
        
        # Initialize mean at the center of the box
        x = np.zeros(n)
        # Initial step size. 
        # Bent Cigar has a minimum at some shifted location. 
        # Starting sigma around 1.0 or 2.0 is reasonable for [-5, 5] box.
        sigma = 1.0
        
        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / (n**1.7 + 10 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n**1.7 + mu_eff))
        
        # Initialize covariance matrix as identity
        C = np.eye(n)
        # Eigenvalues of C (initially 1)
        C_eigvals = np.ones(n)
        # Inverse square root of C
        C_eigvecs = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Update counters
        count_cs = 0
        count_c = 0
        
        # Pre-compute constants
        hsig = 1.0
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Initial population
        # We will manage the budget manually. 
        # CMA-ES usually generates a population of size lambda per generation.
        # Total budget is 500.
        
        # To maximize AOCC, we want to get a good solution early.
        # CMA-ES is stochastic. We can run it in generations.
        
        # Helper to update C's eigen decomposition if needed (approximate for speed in small n)
        # For n=5, full eigendecomposition is very fast.
        
        def get_C_inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical noise
            eigvals = np.clip(eigvals, 1e-12, None)
            inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T, eigvals, eigvecs

        # Initial setup
        C_inv_sqrt, C_eigvals, C_eigvecs = get_C_inv_sqrt(C)
        
        best_x = x.copy()
        best_f = func(x)
        evals = 1
        
        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            Y = np.zeros((lam, n))
            F = np.zeros(lam)
            
            # We might not have budget for a full population if we are close to the limit
            # But usually 500 is divisible by 14 (approx 35 generations).
            # If remaining budget < lam, we just do the remaining single steps? 
            # CMA-ES needs a population. If we can't do a full pop, we stop or do a smaller pop?
            # Standard CMA-ES uses fixed lambda. If budget is not multiple, we might waste evals 
            # or do a partial population. Partial populations are tricky for adaptation.
            # Let's just run full generations. If we have < lam evals left, we break.
            if self.budget - evals < lam:
                break
                
            for i in range(lam):
                # Sample from N(0, C)
                z = np.random.randn(n)
                y = x + sigma * C_inv_sqrt @ z
                # Clip to box [-5, 5]
                # Reflection is better for boundary handling in CMA-ES
                # But simple clipping is often sufficient and simpler. 
                # Let's use clipping as per task description "clips a finite length-5 vector to the box"
                # The func itself clips. But we should keep x inside for stability.
                y = np.clip(y, -5, 5)
                
                Y[i] = y
                F[i] = func(y)
                evals += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()
            
            # Sort population by fitness
            sort_idx = np.argsort(F)
            Y = Y[sort_idx]
            F = F[sort_idx]
            
            # Update mean
            x_new = np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Update path state variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # hsig check (simplified: if ||ps|| is not too large, hsig=1)
            # Full hsig requires eigen decomposition of C. We have it.
            # ||C_inv_sqrt @ (x_new - x) / sigma||^2
            # Actually, the term is C_inv_sqrt @ (x_new - x).
            # Let's use the simplified check or skip hsig for simplicity? 
            # hsig is important for adaptation speed.
            # ||C^{-1/2} (x_new - x)||^2 / sigma^2
            # We computed C_inv_sqrt @ (x_new - x) above for ps update? No, we used it in the formula.
            # Let's recompute or store it.
            # z_mean = C_inv_sqrt @ (x_new - x)
            z_mean = C_inv_sqrt @ (x_new - x)
            norm_z_mean_sq = np.dot(z_mean, z_mean)
            
            # Expected norm of N(0, C)
            # E[||z||^2] = trace(C) = sum(eigvals)
            # The condition is: ||z_mean||^2 / (n * sigma^2) ? No.
            # Standard CMA-ES hsig:
            # hsig = 1 if ||ps||^2 / (1 - (1-c_sigma)^{2*gen}) < (1+2*sqrt((mu_eff-1)/(n+1))) * n + 2
            # We don't track gen count easily here for the exponential term. 
            # Let's use a simpler heuristic or just set hsig=1 if norm is reasonable.
            # For robustness, let's use the norm check.
            # Actually, let's just use hsig=1 for now, or a very simple check.
            # If we skip hsig, c_c adaptation might be slightly less optimal but still works.
            # Let's implement the basic hsig.
            # We need the generation count for the denominator.
            # gen = (evals - 1) // lam
            gen = (evals - 1) // lam
            # This is approximate.
            
            # Let's stick to a robust simplified CMA-ES that doesn't strictly require hsig logic 
            # which is prone to implementation errors, or implement it carefully.
            # Given the short runtime, let's implement hsig properly.
            
            # We need C_inv_sqrt @ (x_new - x) which is z_mean.
            # ps is the path in the C^{-1/2} space? 
            # In standard CMA-ES, ps is in the original space but scaled?
            # Standard formula:
            # ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} (x_new - x) / sigma
            # Wait, the standard definition of ps is often in the "standard normal" space if C=I.
            # With C != I, ps is defined such that its norm is comparable to n.
            
            # Let's use the standard implementation from the CMA-ES website.
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            # This is what I have.
            
            # hsig = 1 if ||ps||^2 / (1 - (1-c_sigma)^{2}) < (1 + 2 * sqrt((mu_eff-1)/(n+1))) * n + 2
            # The denominator is approx 1 for small c_sigma and large gen.
            # Let's just check if ||ps||^2 < (1 + 2 * sqrt((mu_eff-1)/(n+1))) * n + 2
            # This is a sufficient condition for hsig=1 in many cases.
            
            norm_ps_sq = np.dot(ps, ps)
            threshold = (1 + 2 * np.sqrt((mu_eff - 1) / (n + 1))) * n + 2
            hsig = 1.0 if norm_ps_sq < threshold else 0.0
            
            # Update pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update C
            # Delta_hsig = hsig - 1
            delta_hsig = hsig - 1
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * np.outer(z_mean, z_mean), axis=0)
            # Note: z_mean is C^{-1/2} (x_new - x). The rank_mu term uses the sorted z vectors.
            # We need the z vectors for the top mu individuals.
            # Y[:mu] are the sorted y's.
            # z_i = C^{-1/2} (y_i - x) / sigma ?
            # No, the rank_mu update uses the z vectors that generated the y's.
            # y_i = x + sigma * C^{-1/2} z_i
            # So z_i = C^{1/2} (y_i - x) / sigma.
            # We don't have C^{1/2} directly, but we have C_inv_sqrt.
            # z_i = (C_inv_sqrt @ (y_i - x)) / sigma
            # Wait, y_i - x = sigma * C_inv_sqrt @ z_orig
            # So z_orig = C_sqrt @ (y_i - x) / sigma.
            # We don't have C_sqrt.
            # However, rank_mu = sum w_i (z_i - z_mean)(z_i - z_mean)^T ?
            # Standard formula: rank_mu = sum w_i (z_i - z_mean)(z_i - z_mean)^T
            # where z_i are the sampled standard normals? No.
            # In CMA-ES, the rank_mu update is:
            # C = (1 - c_1 - c_mu)C + c_1 (pc pc^T + (1-hsig)c_c(2-c_c)C) + c_mu sum w_i (z_i - z_mean)(z_i - z_mean)^T
            # Here z_i are the vectors such that y_i = x + sigma C^{1/2} z_i.
            # So z_i = C^{-1/2} (y_i - x) / sigma.
            
            # Let's compute z_i for the top mu individuals.
            Z_mu = np.zeros((mu, n))
            for i in range(mu):
                Z_mu[i] = C_inv_sqrt @ (Y[i] - x) / sigma
            
            # z_mean is C_inv_sqrt @ (x_new - x) / sigma
            # Wait, x_new is the weighted average of Y.
            # x_new - x = sum w_i (Y_i - x)
            # C_inv_sqrt @ (x_new - x) / sigma = sum w_i C_inv_sqrt @ (Y_i - x) / sigma = sum w_i Z_mu_i
            # So z_mean_computed = sum w_i Z_mu[i]
            # We already computed z_mean = C_inv_sqrt @ (x_new - x).
            # So z_mean_scaled = z_mean / sigma.
            
            z_mean_scaled = z_mean / sigma
            rank_mu = np.sum(weights[:, None] * np.outer(Z_mu - z_mean_scaled[None, :], Z_mu - z_mean_scaled[None, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update C_inv_sqrt
            C_inv_sqrt, C_eigvals, C_eigvecs = get_C_inv_sqrt(C)
            
            x = x_new
            
            # Ensure x is in bounds (though it's a mean, it can be outside, but func clips)
            # It's better to keep x in bounds for numerical stability?
            # CMA-ES handles boundaries via the sampling. The mean can drift outside.
            # But if mean is far outside, sampling might be inefficient.
            # Clipping the mean is a heuristic. Let's not clip the mean to avoid biasing the CMA-ES dynamics.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
