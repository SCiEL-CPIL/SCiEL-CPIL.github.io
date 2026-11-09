"""Rewritten optimizer for BBOB f6 (Attractive sector) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Continuous optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        x_best = None
        f_best = np.inf

        def eval_func(x):
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        # CMA-ES Parameters
        popsize = 10  # Lambda
        max_generations = budget // popsize
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Strategy parameters
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = np.min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2) ** 2 + mu_eff))

        # Main loop
        for gen in range(max_generations):
            if calls_made >= budget:
                break

            # Generate offspring
            Y = np.zeros((popsize, dim))
            X = np.zeros((popsize, dim))
            F = np.zeros(popsize)
            
            for i in range(popsize):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                y = np.linalg.cholesky(C) @ z
                x = x_mean + sigma * y
                # Handle boundaries: reflect if out of bounds
                # This is a simple heuristic. More advanced CMA-ES handles this differently,
                # but for [-5,5] and dim 5, reflection is robust.
                for d in range(dim):
                    if x[d] < -5:
                        x[d] = -5 + (-5 - x[d]) * 0.5 # Damped reflection
                        if x[d] > 5: x[d] = 5
                    elif x[d] > 5:
                        x[d] = 5 - (x[d] - 5) * 0.5
                        if x[d] < -5: x[d] = -5
                
                Y[i] = y
                X[i] = x
                F[i] = eval_func(x)

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            X = X[order]
            F = F[order]

            if calls_made >= budget:
                break

            # Update mean
            x_mean_new = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update covariance matrix
            # Weighted outer product of differences
            v = (X[:mu] - x_mean) / sigma
            delta_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))
            c_sigma_factor = c_sigma * (1 - c_sigma) * (delta_sigma > 1.4 + 2 / (dim + 1))
            
            # Update ps and pc
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ pc) # Wait, standard update is:
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) / sigma * (x_mean_new - x_mean)
            # Actually, the vector added is the step in normalized space.
            step = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step
            
            # pc update
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 + c_sigma) / (1 - c_sigma)))
            pc = (1 - c_c) * pc + h_sigma * c_c * step
            
            # C update
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C # Decay
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (h_sigma and (1 - h_sigma) or 0) * (c_c * (2 - c_c) * C - np.outer(pc, pc))) 
            # The above C update formula is complex. Let's use the standard simpler form often found in implementations:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc^2 + (1-h_sigma)*c_c*(2-c_c)*C - c_c*pc^2) + c_mu * sum(w_i * (y_i - y_bar)^2)
            
            # Let's stick to a robust standard implementation structure.
            # Recalculate C update properly:
            C_decay = 1 - c_1 - c_mu
            C = C_decay * C
            C += c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * C - c_c * np.outer(pc, pc))
            
            # Add weighted sum of outer products of selected steps
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(v[i], v[i])

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            
            # Update mean
            x_mean = x_mean_new

            # Check for convergence
            if np.linalg.norm(x_mean - x_mean_new) < 1e-6 * (1 + np.abs(x_mean)):
                 # Optional: break if stuck, but we have budget
                 pass

        return x_best, f_best
# EVOLVE-BLOCK-END
