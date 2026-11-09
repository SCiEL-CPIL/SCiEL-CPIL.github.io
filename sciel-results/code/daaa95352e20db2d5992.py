import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # CMA-ES Parameters
        # For dim=5, lambda=10 is standard.
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        # Start near center, as minimizer is in [-4, 4]^5
        # Initial sigma should cover the box reasonably well.
        # Knowledge: "slightly larger initial sigma (2.5) improves exploration"
        x0 = np.zeros(dim)
        sigma = 2.5
        
        # CMA-ES State
        C = np.eye(dim)
        Cs = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Step sizes
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2 / 3) - 2) / ((dim + 2) ** 2 + mueff))
        
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial population
        pop = np.zeros((lam, dim))
        fit = np.zeros(lam)
        
        for i in range(lam):
            if calls_used >= budget:
                break
            pop[i] = x0 + sigma * np.random.randn(dim)
            fit[i] = track_call(pop[i])
            
        # Track best
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # Main Loop
        while calls_used < budget and time.time() < deadline - 2.0:
            # Sort population by fitness
            order = np.argsort(fit)
            y = pop[order]
            y_fit = fit[order]
            
            # Weighted mean
            m = np.dot(weights[:mu], y[:mu, :])
            
            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (m - x0)
            
            # Hoeffding's bound
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lam + 1))) > (chiN + 1.4) * 1.1:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Update evolution path for covariance
            Cs = (1 - c_c) * Cs + c_c * (m - x0) / sigma
            
            # Rank-1 update
            C = (1 + c_1 * (c_mu + c_1) - c_1) * C + c_1 * np.outer(Cs, Cs)
            
            # Rank-mu update
            if c_mu > 0:
                # Standard CMA-ES rank-mu update uses (y_i - m)
                # Note: y is sorted, so y[:mu] are the best
                diff = (y[:mu, :] - m[None, :]) / sigma
                C += c_mu * np.sum(weights[:mu, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            # For dim=5, this is usually stable, but let's be safe
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update mean
            x0 = m
            
            # Generate new population
            new_pop = np.zeros((lam, dim))
            new_fit = np.zeros(lam)
            
            for i in range(lam):
                if calls_used >= budget:
                    break
                
                # Sample from multivariate normal
                # Use Cholesky decomposition for efficiency and stability
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    L = np.eye(dim)
                
                z = np.random.randn(dim)
                new_pop[i] = x0 + sigma * (L @ z)
                
                # Error Repair Strategy:
                # Instead of clipping or resampling, we "repair" out-of-bound points.
                # This preserves the direction of the step but constrains the magnitude.
                # If a point is out of bounds, we project it back into the box.
                # Simple clipping is a form of repair, but we can be smarter.
                # For now, clipping is the most robust "repair" that doesn't waste evaluations.
                # The knowledge says "resampling prevents wasting... but slightly larger sigma improves".
                # Let's stick to clipping as it's equivalent to reflecting for the purpose of staying in bounds,
                # but the key is that we evaluate the point at the boundary rather than discarding it.
                
                # Actually, the prompt says "func(x) clips... and returns a float".
                # So the function itself clips.
                # The "error repair" idea target asks: Does operator 'error_repair' change the chance...
                # In CMA-ES context, "error repair" often refers to handling infeasible points.
                # Since the function clips, any point is "feasible" in terms of evaluation.
                # However, points far outside the box provide no gradient info.
                # We can implement a "reflection" or "resampling" if we want to be strict.
                # But the function clips, so evaluating x=[10, 10, 10, 10, 10] gives f(5, 5, 5, 5, 5).
                # This is effectively the same as clipping.
                # The "repair" here is implicit in the function.
                # To make it explicit and potentially better, we can resample if the point is far outside.
                # But resampling wastes the evaluation if we don't call func.
                # If we resample, we don't call func, so we save budget? No, we still need to evaluate the new point.
                # So resampling just changes the point we evaluate.
                # Let's try a simple reflection: if x > 5, x = 5 - (x - 5) = 10 - x.
                # This keeps the point inside the box and preserves some directional info.
                
                # Let's implement reflection as the "error repair" mechanism.
                # This is different from clipping (which saturates) or resampling (which loses direction).
                # Reflection:
                # x_reflected = 2 * bound - x if x > bound or x < -bound
                # For x > 5: 10 - x
                # For x < -5: -10 - x
                
                # Let's apply reflection to keep points within [-5, 5]
                # Note: Multiple reflections might be needed if x is very far out, but sigma starts at 2.5 and shrinks, so unlikely to be > 10 away.
                # Simple single reflection:
                mask_high = new_pop[i] > 5
                mask_low = new_pop[i] < -5
                new_pop[i][mask_high] = 10 - new_pop[i][mask_high]
                new_pop[i][mask_low] = -10 - new_pop[i][mask_low]
                
                # Ensure it's within bounds after reflection (in case of numerical issues)
                new_pop[i] = np.clip(new_pop[i], -5, 5)
                
                new_fit[i] = track_call(new_pop[i])
            
            pop = new_pop
            fit = new_fit
            
            # Update best
            current_best_idx = np.argmin(fit)
            if fit[current_best_idx] < best_f:
                best_f = fit[current_best_idx]
                best_x = pop[current_best_idx].copy()

        return best_x, best_f
