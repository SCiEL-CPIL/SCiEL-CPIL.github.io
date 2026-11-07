"""Combined CMA-ES with L-BFGS-B polish for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive covariance followed by L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Reserve budget for L-BFGS-B polish
        # L-BFGS-B typically converges quickly on smooth functions.
        # 50 calls should be enough for a few iterations with finite diff gradients.
        polish_budget = min(50, budget // 10)
        es_budget = budget - polish_budget

        # Initial point: Random uniform in [-5, 5]
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # CMA-ES Parameters
        # Standard CMA-ES parameters for dim=5
        # Lambda (population size)
        lambda_ = 4 + int(3 * np.log(dim))
        
        # Initial sigma
        # Bent Cigar has a wide valley. Starting with a larger sigma helps escape local flatness or find the valley direction faster.
        # However, too large might overshoot. 1.0 is a standard start.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # CMA-ES Loop
        while calls_made < es_budget and time.time() < deadline:
            # Generate lambda_ candidates
            # We need to ensure we don't exceed budget. 
            # If remaining budget < lambda_, we break or reduce lambda_.
            # For simplicity and safety, if we can't afford a full generation, we break.
            if es_budget - calls_made < lambda_:
                break
                
            z = rng.standard_normal((lambda_, dim))
            
            # Transform z using C and sigma
            # y = x + sigma * C^{1/2} z
            # We can use Cholesky or SVD. Cholesky is faster if C is PD.
            # To ensure PD, we might need to add small regularization if it fails, but CMA-ES updates usually keep it PD.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to SVD if Cholesky fails
                U, s, Vt = np.linalg.svd(C)
                L = (Vt.T * np.sqrt(s)).T
            
            ys = np.einsum('ij,kj->ki', L, z) * sigma
            ys += x
            
            # Clip to box? 
            # The task says func clips. So we can pass unclipped ys to func, 
            # but for CMA-ES statistics, using the clipped points might be better 
            # if the boundary is important. However, Bent Cigar is smooth and 
            # minimum is inside [-4,4]^5. 
            # Standard CMA-ES doesn't clip. We rely on func's clipping.
            
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                fvals[i] = func(ys[i])
                calls_made += 1
                
                # Update best
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = ys[i].copy()
                
                if time.time() > deadline:
                    break
            
            # Check if we ran out of time or budget mid-generation
            if calls_made >= es_budget or time.time() >= deadline:
                break
                
            # Sort by fitness
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]
            
            # Compute mean
            x_new = np.sum(ys[:mu] * weights[:, None], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (x_new - x) / sigma
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * (x_new - x) / sigma
            
            # We need C^{-1/2} and C^{1/2}. We have L such that C = L L^T.
            # C^{-1/2} is not just L^{-1}. 
            # Let's recompute Cholesky for the update if needed, or use the one from generation.
            # The L used for generation is for the OLD C.
            
            # Compute C^{1/2} and C^{-1/2} from current C (before update)
            try:
                L_curr = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L_curr)
                # C^{-1/2} = L^{-T} ? No.
                # If C = L L^T, then C^{-1/2} is not simply related to L.
                # However, for the path updates, we usually use the square root of the covariance.
                # A common approximation or exact method involves diagonalizing or using the Cholesky factor.
                # Let's use the property that if we generate z ~ N(0, I), y = L z ~ N(0, C).
                # The term (x_new - x) / sigma is a vector in "standard normal" space scaled by sigma.
                # To map it back to "C-space" for p_sigma, we need to apply C^{-1/2}.
                
                # Simpler approach for small dim:
                # p_sigma update:
                # delta = (x_new - x) / sigma
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ delta
                
                # We can compute C_inv_sqrt via eigen decomposition or SVD for robustness in dim=5.
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-12, None) # Ensure positive
                C_inv_sqrt = V @ np.diag(1/np.sqrt(w)) @ V.T
                
            except Exception:
                C_inv_sqrt = np.eye(dim)
                
            delta = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta)
            
            # h_sigma check (optional but good for stability)
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(dim+1)) * E||N(0,I)||
            # For simplicity and speed, we often skip h_sigma in compact implementations if parameters are standard.
            # Let's include a simple check or just proceed.
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta
            
            # Update Covariance
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i - x)(y_i - x)^T
            # Note: y_i - x should be the step vector.
            # The y_i stored are the actual points. The steps are y_i - x.
            # But wait, the y_i were generated as x + sigma * L z.
            # So y_i - x = sigma * L z_i.
            # The outer product is (y_i - x)(y_i - x)^T.
            
            h_sigma = 1.0 # Simplified
            
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] * np.einsum('i,jk,il->ikl', np.zeros((mu, dim)), 
                         (ys[:mu] - x)[:, None, :], (ys[:mu] - x)[:, None, :]), axis=0)
            # The einsum above is messy. Let's do it simply:
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = ys[i] - x
                rank_mu += weights[i] * np.outer(diff, diff)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| for dim=5 is approx 2.13
            c_sigma_factor = c_sigma / d_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            e_norm = 2.13 # Approx for dim=5
            sigma *= np.exp(c_sigma_factor * (norm_p_sigma / e_norm - 1.0))
            
            # Clamp sigma
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-12:
                sigma = 1e-12
            
            x = x_new
            fx = fvals[0] # The best of the generation

        # Phase 2: L-BFGS-B Polish
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            lbfgs_calls = [0]
            
            def lbfgs_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if lbfgs_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                lbfgs_calls[0] += 1
                z_clipped = np.clip(z, -5, 5)
                return func(z_clipped)
            
            try:
                result = minimize(
                    lbfgs_func,
                    best_x,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 100, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
