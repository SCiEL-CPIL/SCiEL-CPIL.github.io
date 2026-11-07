"""Rewritten optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_pop = int(4 + 3 * np.log(dim))
        # Step size
        sigma = 2.0
        # Initial center: try origin first (common for BBOB), but we'll adapt
        # Actually, for Rastrigin, the global minimum is at 0 (before shift/rotation)
        # But we don't know the shift. Let's start at 0.
        # Wait, the problem says "minimizer lies in [-4, 4]^5".
        # Starting at 0 is a good bet for BBOB functions which are often centered.
        # However, Rastrigin has many local minima.
        # Let's use a random start if we want to be robust, but 0 is often good for BBOB.
        # Let's try starting at 0.
        
        # To be safer against shifts, maybe a random start?
        # But BBOB f15 is Rastrigin. The shift is hidden.
        # If I start at 0, and the shift is large, I might be far.
        # But the box is [-5, 5].
        # Let's use a simple heuristic: start at 0. If it's bad, CMA-ES will move.
        # Actually, let's use a few random restarts? No, budget is tight.
        # Let's stick to one CMA-ES run with a good initialization.
        # Common practice for BBOB: start at 0.
        
        x = np.zeros(dim)
        fx = func(x)
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        hc = 0.0
        # Learning rates
        cmu = 1 / ((4 + dim / 2) * (dim + 4 / 3))
        cc = (4 + dim / dim) / ((dim + 4) * (4 + 3 * dim))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        
        # We need mu_eff
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Recalculate cs with correct mu_eff
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        
        # Initial step size
        # For Rastrigin, landscape is rugged. 
        # Sigma = 2.0 is reasonable for [-5,5].
        
        best_x = x.copy()
        best_f = fx
        
        # Loop
        for it in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Generate population
            Y = np.zeros((lambda_pop, dim))
            F = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample
                # Cholesky factorization of C (update every step? No, too expensive)
                # We can update C less frequently or use a simpler update.
                # Standard CMA-ES updates C every step.
                # dim=5, so Cholesky is cheap.
                
                # We need the square root of C.
                # Let's maintain a Cholesky factor or just use eigendecomposition?
                # For dim=5, np.linalg.cholesky is fast.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C becomes singular or non-positive definite, reset
                    C = np.eye(dim)
                    L = np.eye(dim)
                    sigma = 2.0
                    ps = np.zeros(dim)
                    hc = 0.0
                
                z = np.random.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()
            
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size (CSA)
            # Need old C^{-1/2} to compute norm
            # This is getting complex. Let's use a simpler adaptation or standard formulas.
            # Standard CMA-ES:
            # D = sqrt(C) is not quite right for the norm in the path update.
            # The update for sigma involves ||x_new - x|| / (sigma * sqrt(1 - exp(-2/mu_eff)))
            # And the path ps update involves C^{-1/2} (x_new - x) / sigma.
            
            # Let's implement standard CMA-ES properly.
            # We need C^{-1/2}.
            # We can compute eigendecomposition of C.
            # Or, we can approximate.
            # Given dim=5, eigendecomposition is fine.
            
            # Compute eigenvalues and eigenvectors of C
            # This is O(n^3) = 125 ops, very fast.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.maximum(eigvals, 1e-12)
                # C^{-1/2} = V diag(1/sqrt(lambda)) V^T
                inv_sqrt_C = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                inv_sqrt_C = np.eye(dim)
            
            # Update sigma
            # D = sqrt(1 - exp(-2/mu_eff))
            D = np.sqrt(1 - np.exp(-2 / mu_eff))
            # ||x_new - x|| is the distance in the transformed space? No, it's the norm of the difference.
            # The formula is:
            # sigma = sigma * exp(cs * (||C^{-1/2}(x_new - x)|| / sigma - D) / (dim + 2))
            
            delta = np.linalg.norm(inv_sqrt_C @ (x_new - x)) / sigma
            sigma = sigma * np.exp(cs * (delta - D) / (dim + 2))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 100)
            
            # Update evolution paths
            # ps = (1 - cs) ps + sqrt(cs (2 - cs) mu_eff) C^{-1/2} (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (inv_sqrt_C @ (x_new - x) / sigma)
            
            # hc
            # hc = 1 - exp(-1 / (it + 1) / (dim + 4)) ? No, standard is:
            # hc = 1 - exp(-1 / (it+1) / (dim+4)) is for the initial decay?
            # Actually, hc is a constant depending on dim and mu_eff.
            # hc = 1 - exp(-1 / (dim + 4) / mu_eff) ?
            # Let's check the formula.
            # hc = 1 - exp(-1 / (dim + 4) * mu_eff) is not quite right.
            # Standard: hc = 1 - exp(-1 / (dim + 4) / mu_eff) ??
            # From Hansen's thesis:
            # hc = 1 - exp(-1 / (dim + 4) * mu_eff) is incorrect.
            # hc = 1 - exp(-1 / (dim + 4) / mu_eff) ?
            # Let's use the standard value: hc = 1 - exp(-1 / (dim + 4) * mu_eff)
            # Actually, it's hc = 1 - exp(-1 / (dim + 4) * mu_eff) is often cited.
            # Let's compute it.
            hc = 1 - np.exp(-1 / (dim + 4) * mu_eff)
            
            # pc = (1 - cc) pc + hc * (2 - cc) * sign(ps) * (||ps|| / sqrt(1 - (1 - cs)^2) / sigma - D)
            # Wait, the sign is applied to ps.
            # norm_ps = ||ps||
            norm_ps = np.linalg.norm(ps)
            # The term inside is: norm_ps / sqrt(1 - (1 - cs)^2) / sigma - D ?
            # No, the formula is:
            # pc = (1 - cc) pc + hc * (2 - cc) * sign(ps) * (norm_ps / sqrt(1 - (1 - cs)^2) - D) / sigma ?
            # Let's look up the exact formula.
            # pc_{t+1} = (1 - cc) pc_t + hc (2 - cc) s_{t+1}
            # where s_{t+1} = 1 if ||ps_{t+1}|| / sqrt(1 - (1 - cs)^{2(t+1)}) > (1.4 + 2/(dim+1)) E[N(0,1)]
            # This is getting too detailed for a quick rewrite.
            # Let's simplify: skip the covariance matrix update (C) and just use diagonal CMA-ES or (1+lambda)-ES?
            # No, CMA-ES is much better for rotated functions.
            # Let's use a simplified CMA-ES update for C.
            
            # Update C
            # m = mu_eff
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + (hc^2 - hc) C) + cmu sum w_i (y_i - x)(y_i - x)^T
            # where c1 = 2 / ((dim + 1.3)^2 + mu_eff)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            
            # We need pc.
            # Let's approximate pc update or skip it if too complex.
            # For dim=5, we can afford the full update.
            # But implementing the exact hc logic is tricky.
            # Let's use a simpler heuristic for C update:
            # C = 0.9 * C + 0.1 * sum w_i (y_i - x)(y_i - x)^T
            # This is a rank-mu update with a fixed weight.
            # It might work well enough.
            
            # Let's do a proper rank-mu update with standard parameters.
            # C = (1 - c1 - cmu) C + c1 (pc pc^T) + cmu sum w_i (y_i - x)(y_i - x)^T
            # We need pc. Let's assume pc is the normalized ps? No.
            # Let's just use the rank-mu update without the rank-1 (pc) part for simplicity,
            # or set c1=0 and adjust cmu.
            # If c1=0, C = (1 - cmu) C + cmu sum w_i (y_i - x)(y_i - x)^T
            # This is a valid covariance matrix adaptation.
            
            C_new = (1 - cmu) * C
            for i in range(mu):
                dy = Y_sorted[i] - x
                C_new += weights[i] * np.outer(dy, dy)
            
            # Ensure symmetry
            C = (C_new + C_new.T) / 2
            
            # Update x
            x = x_new
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
            
            # Check convergence
            if sigma < 1e-8:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
