"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Solver using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    Strategy:
    1. Brief initial exploration to find a promising starting point.
    2. Run CMA-ES which adapts the full covariance matrix to handle 
       the ill-conditioned and rotated landscape of Schaffers F7.
    3. This allows the search distribution to align with the function's 
       valley, leading to faster convergence than diagonal-only methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        rng = np.random.default_rng(42)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Initial Exploration ---
        # Use a small number of random points to initialize the CMA-ES mean.
        # 10 points is sufficient to get a rough idea of the basin.
        n_init = 10
        init_points = rng.uniform(-5, 5, size=(n_init, dim))
        init_scores = np.full(n_init, np.inf)
        
        for i in range(n_init):
            if time.time() > deadline or calls >= budget:
                break
            init_scores[i] = counting_func(init_points[i])
        
        if calls == 0:
             # Fallback if something went wrong immediately
             return best_x, best_f

        # Find the best initial point
        best_idx = np.argmin(init_scores)
        x_mean = init_points[best_idx].copy()
        f_best_init = init_scores[best_idx]
        
        # --- Phase 2: CMA-ES ---
        # Parameters for CMA-ES
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))
        # mu: number of points used for recombination
        mu = lam // 2
        # weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # equivalent sample size
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky factor of C
        B = np.linalg.cholesky(C)
        # D = 1/sqrt(diag(C))
        D = 1.0 / np.sqrt(np.diag(C))
        
        # Path evolution strategies
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Damping for step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)
        
        # Stagnation counter
        stagnation = 0
        max_stagnation = 20
        
        while calls < budget and time.time() < deadline:
            # Generate offspring
            z = rng.standard_normal((lam, dim))
            # Transform z to x
            # x = x_mean + sigma * B * z
            # B is (dim, dim), z is (lam, dim) -> need z @ B.T
            offspring = x_mean[None, :] + sigma * (z @ B.T)
            
            # Evaluate offspring
            offspring_f = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                offspring_f[i] = counting_func(offspring[i])
            
            if np.any(np.isinf(offspring_f)):
                break

            # Sort offspring by fitness
            sorted_indices = np.argsort(offspring_f)
            sorted_offspring = offspring[sorted_indices]
            sorted_f = offspring_f[sorted_indices]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
                stagnation = 0
            else:
                stagnation += 1
                
            # Stop if stagnating too long and sigma is small
            if stagnation > max_stagnation and sigma < 0.1:
                break

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (sorted_offspring[i] - x_mean)
            y_w /= sigma
            x_mean = x_mean + sigma * y_w
            
            # Update path evolution strategies
            inv_sqrt_C = np.linalg.inv(np.diag(D)) # Approximation, actually inv(sqrt(C))
            # More accurate: B_inv = inv(B), D is diag. 
            # Standard formula: p_sigma += c_sigma/sqrt(c_sigma) * (sqrt(C) y_w - p_sigma)
            # sqrt(C) = B @ diag(D) is not quite right. 
            # Let's use the standard update rules.
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mueff) * (1/sqrt(C)) y_w
            # 1/sqrt(C) is applied to y_w.
            # y_w is in "standard normal" space scaled by sigma.
            # We need C^{-1/2} y_w.
            # Since C = B B^T, C^{-1/2} = B^{-T} B^{-1}.
            # This is computationally expensive. 
            # Standard CMA-ES implementations often use the update:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mueff) * (B^{-1} y_w / D) ? No.
            
            # Let's stick to the standard formulas from Hansen's paper.
            # y_w is the weighted sum of (x_i - x_mean)/sigma.
            # p_sigma update:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mueff) * (1/sqrt(C)) y_w
            # To compute (1/sqrt(C)) y_w, we can solve B^T v = y_w / D ?
            # Actually, y_w is already in the transformed space if we defined it correctly.
            # Let's re-verify the transformation.
            # x_i = x_mean + sigma * B * z_i
            # y_i = (x_i - x_mean)/sigma = B * z_i
            # y_w = sum w_i y_i = B * (sum w_i z_i)
            # So y_w is in the "z-space" transformed by B.
            # The update for p_sigma uses C^{-1/2} y_w.
            # C^{-1/2} = B^{-T} B^{-1} is wrong. C = B B^T. C^{-1/2} = B^{-T} B^{-1} only if B is symmetric? No.
            # C^{-1/2} y_w = B^{-T} (B^{-1} y_w)?
            # Let's use the property that if y = B z, then C^{-1/2} y = z.
            # So C^{-1/2} y_w = sum w_i z_i.
            
            # Let's recompute z_w = sum w_i z_sorted[i]
            z_sorted = z[sorted_indices]
            z_w = np.zeros(dim)
            for i in range(mu):
                z_w += weights[i] * z_sorted[i]
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_w
            
            # Update p_c
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mueff) * C^{-1/2} y_w
            # C^{-1/2} y_w = z_w
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(gen+1)}) < (1 + 2/(dim+1)) chi_n
            
            # We need to track the generation or use a simpler heuristic for h_sigma.
            # For simplicity and robustness, we can set h_sigma = 1 if ||p_sigma|| < 1.4 * chi_n else 0.
            # Or better, calculate the threshold.
            # Since we don't track 'gen' explicitly in this loop structure easily without extra state, 
            # we can approximate h_sigma.
            norm_p_sigma = np.linalg.norm(p_sigma)
            # A common simplification is to check if norm_p_sigma is small.
            h_sigma = 1.0 if norm_p_sigma < 1.4 * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * z_w
            
            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
            # Note: y_i y_i^T = (B z_i) (B z_i)^T = B (z_i z_i^T) B^T
            
            # Rank-one update
            y_w_matrix = np.outer(y_w, y_w)
            # Rank-mu update
            y_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_mu += weights[i] * np.outer(y_w_sorted[i] if False else (sorted_offspring[i] - x_mean)/sigma, (sorted_offspring[i] - x_mean)/sigma)
            
            # Wait, y_w in the formula is the weighted mean vector.
            # The rank-mu update uses the individual y_i vectors.
            # Let's compute y_sorted = (sorted_offspring - x_mean_old) / sigma
            # But x_mean has already been updated. We need the old x_mean.
            
            # Let's refactor to keep x_mean_old.
            
            # RESTART CMA-ES LOGIC WITH PROPER STATE TRACKING
            break # Break out of this flawed loop implementation and restart properly below
            
        # The previous loop had issues with state management (x_mean updated before rank-mu calc).
        # Let's implement a cleaner version.
        
        # --- Phase 2 (Corrected): CMA-ES ---
        
        # Reset state if we broke out early due to logic error, or just use the best_x found so far
        # If we used up budget in Phase 1, return.
        if calls >= budget:
            return best_x, best_f
            
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.linalg.cholesky(C)
        D = 1.0 / np.sqrt(np.diag(C))
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)
        
        # Re-define parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        stagnation = 0
        max_stagnation = 20
        
        while calls < budget and time.time() < deadline:
            # Generate offspring
            z = rng.standard_normal((lam, dim))
            offspring = x_mean[None, :] + sigma * (z @ B.T)
            
            # Evaluate offspring
            offspring_f = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                offspring_f[i] = counting_func(offspring[i])
            
            if np.any(np.isinf(offspring_f)):
                break

            # Sort offspring
            sorted_indices = np.argsort(offspring_f)
            sorted_offspring = offspring[sorted_indices]
            sorted_f = offspring_f[sorted_indices]
            sorted_z = z[sorted_indices]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
                stagnation = 0
            else:
                stagnation += 1
                
            if stagnation > max_stagnation and sigma < 0.05:
                break

            # Save old mean for y calculation
            x_mean_old = x_mean.copy()
            
            # Calculate y vectors: y_i = (x_i - x_mean_old) / sigma
            y_sorted = (sorted_offspring - x_mean_old[None, :]) / sigma
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * y_sorted[i]
            x_mean = x_mean_old + sigma * y_w
            
            # Update p_sigma
            # C^{-1/2} y_w = z_w (since y_w = B z_w)
            z_w = np.zeros(dim)
            for i in range(mu):
                z_w += weights[i] * sorted_z[i]
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_w
            
            # Update p_c
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < 1.4 * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * z_w
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
            
            # Rank-one part
            p_c_outer = np.outer(p_c, p_c)
            
            # Rank-mu part
            y_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_mu += weights[i] * np.outer(y_sorted[i], y_sorted[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c_outer + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * y_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            
            # Update B and D
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add small identity
                C += 1e-6 * np.eye(dim)
                B = np.linalg.cholesky(C)
            
            D = 1.0 / np.sqrt(np.diag(C))
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 5.0)

        # --- Phase 3: Final Polish (Optional) ---
        # If we have budget left, do a simple local search
        if calls < budget and time.time() < deadline:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_polish = 0.1
            
            while calls < budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                x_candidate = x_polish + sigma_polish * z
                x_candidate = np.clip(x_candidate, -5, 5)
                
                f_candidate = counting_func(x_candidate)
                
                if f_candidate < f_polish:
                    x_polish = x_candidate
                    f_polish = f_candidate
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.9
                
                if sigma_polish < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
