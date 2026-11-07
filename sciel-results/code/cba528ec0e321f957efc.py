"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility across instances if needed, 
        # but random initialization is better for exploration.
        # The task says "hidden seeded shift", so random init is safe.
        # Let's use a deterministic seed for the RNG to ensure consistent behavior 
        # but different starting points? No, the task implies we just need to minimize.
        # Using random seed is fine.
        rng = np.random.default_rng()

        # Initialize center
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        # Initialize covariance matrix (identity) and step size
        sigma = 1.0
        C = np.eye(dim)
        
        # Eigen-decomposition cache
        D_inv = np.ones(dim)
        B = np.eye(dim)
        
        # Learning rates
        c_sigma = (1 + 0.3 * min(1, (dim - 1) / dim)) / (dim + 3)
        c_c = (4 + dim / dim) / (dim + 4 + 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + mu_eff) - 1))
        
        mu = dim // 2 + 1  # number of recombination weights
        # Calculate weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Update c_mu with correct mu_eff
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + mu_eff) - 1))
        
        # Initial parameters
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Budget management
        # We need to be careful with the number of function evaluations.
        # Each iteration uses 'mu' evaluations.
        # We will run as many full iterations as possible.
        
        calls_made = 1
        remaining = budget - 1
        
        while remaining >= mu:
            # Generate offspring
            ys = np.zeros((mu, dim))
            fvals = np.zeros(mu)
            
            for i in range(mu):
                # Sample y = x + sigma * B * D * z
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D_inv * z))
                # Clip to bounds? The problem says func clips. 
                # But for CMA-ES internal state, clipping y might distort the search.
                # However, the contract says "func(x) clips ... and returns a float".
                # So we can pass unclipped y, and func will clip it.
                # But if y is far outside, the gradient info might be poor.
                # Let's pass y directly.
                y = np.clip(y, -5, 5) # Optional: clip before passing to keep x within bounds logic consistent
                # Actually, better to let func clip, but keep x within bounds for stability.
                # If x is clipped, the distribution might get stuck at boundaries.
                # Let's clip x as well.
                y = np.clip(y, -5, 5)
                fvals[i] = func(y)
                ys[i] = y
                calls_made += 1
            
            remaining -= mu
            
            # Sort by fitness
            idx = np.argsort(fvals)
            ys_sorted = ys[idx]
            fvals_sorted = fvals[idx]
            
            # Update x
            x_new = np.sum(w[:, None] * ys_sorted, axis=0)
            
            # Check for improvement
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                # We don't update x to the best immediately in standard CMA-ES,
                # x is the weighted average. But we can track best found.
                # The return value is ignored, but we want to minimize.
                # The evaluator tracks best value after call i.
                # So we just need to make good calls.
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * (C^{-1/2} (x_new - x))
            # C^{-1/2} = B D
            d = (x_new - x) / sigma
            # C^{-1/2} d = B D (B^T d) ? No.
            # If C = B D^2 B^T, then C^{-1/2} = B D B^T is wrong.
            # C^{-1/2} = B diag(1/D) B^T is not correct either.
            # Actually, C^{-1/2} = B D^{-1} B^T is for C = B D^2 B^T?
            # Let's use the standard formula:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * (C^{-1/2} (x_new - x)) / sigma
            # C^{-1/2} = B D^{-1} B^T is incorrect.
            # C = B D^2 B^T. C^{-1} = B D^{-2} B^T.
            # C^{-1/2} = B D^{-1} B^T is only true if B is orthogonal and D is diagonal.
            # Yes, B is orthogonal. So C^{-1/2} = B D^{-1} B^T.
            # Wait, (B D^{-1} B^T) (B D^2 B^T) = B D^{-1} (B^T B) D^2 B^T = B D D^2 B^T != I.
            # Correction: C^{-1/2} = B D^{-1} B^T is wrong.
            # Let C = B D^2 B^T. Then C^{-1} = B D^{-2} B^T.
            # C^{-1/2} = B D^{-1} B^T is correct because (B D^{-1} B^T)^2 = B D^{-2} B^T B D^{-1} B^T = B D^{-2} D^{-1} B^T? No.
            # (B D^{-1} B^T) (B D^{-1} B^T) = B D^{-1} (B^T B) D^{-1} B^T = B D^{-2} B^T = C^{-1}.
            # So yes, C^{-1/2} = B D^{-1} B^T.
            
            # Vector v = C^{-1/2} (x_new - x)
            # v = B D^{-1} B^T (x_new - x)
            # Let u = B^T (x_new - x)
            # v = B (D^{-1} u)
            
            x_diff = x_new - x
            u = B.T @ x_diff
            v = B @ (u / D_inv)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            
            # Update h_sigma
            h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made))) < (1.4 + 2 / (dim + 1)) * chi_n) else 0.0
            
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_diff / sigma)
            
            # Update covariance matrix C
            # Delta H = c_1 (pc pc^T - (1 - h_sigma^2)(1 - c_c) C) + c_mu sum w_i (y_i - x)(y_i - x)^T
            # Note: y_i - x is the step from old x to offspring.
            
            # Compute rank-1 update from pc
            pc_outer = np.outer(pc, pc)
            
            # Compute rank-mu update
            # ys_sorted - x
            steps = ys_sorted - x
            # Weighted sum of outer products
            # w_i * steps_i steps_i^T
            # Efficient way: (steps * w[:, None])^T steps
            rank_mu = (steps.T * w) @ steps
            
            # Update C
            C = (1 - (c_1 + c_mu) + h_sigma * (1 - h_sigma) * c_c * (2 - c_c) + c_1 * (1 - h_sigma)) * C \
                + c_1 * pc_outer + c_mu * rank_mu
            
            # Wait, the formula for C update is:
            # C = (1 - c_1 - c_mu + c_1 * h_sigma * (1 - h_sigma) * c_c * (2 - c_c)) * C + c_1 * pc pc^T + c_mu * sum ...
            # Let's stick to a simpler, more stable variant if the above is complex.
            # Standard CMA-ES update:
            # C = (1 - c_1 - c_mu + c_1 * h_sigma * (1 - h_sigma) * c_c * (2 - c_c)) * C + c_1 * pc pc^T + c_mu * sum(w_i * (y_i - x)(y_i - x)^T)
            
            # Recalculate C with correct formula
            factor = 1 - c_1 - c_mu + c_1 * h_sigma * (1 - h_sigma) * c_c * (2 - c_c)
            C = factor * C + c_1 * pc_outer + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigen-decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.clip(eigvals, 1e-12, None)
                D_inv = 1.0 / np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                # Fallback to identity if decomposition fails
                C = np.eye(dim)
                D_inv = np.ones(dim)
                B = np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update x
            x = x_new
            
            # Clip x to bounds
            x = np.clip(x, -5, 5)
            
            # If we are close to the end, we can do single steps
            if remaining > 0 and remaining < mu:
                # Do single steps with current sigma and C
                for _ in range(remaining):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D_inv * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    calls_made += 1
                break
        
        return x, fx
# EVOLVE-BLOCK-END
