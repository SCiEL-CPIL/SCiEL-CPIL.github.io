import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial state
        m = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        
        # Adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = 0
        best_x = m.copy()
        best_f = float('inf')
        
        # Initial generation
        pop = m + sigma * rng.multivariate_normal(np.zeros(dim), C, lam)
        pop = np.clip(pop, -5, 5)
        fit = np.array([func(x) for x in pop])
        calls_made += lam
        
        order = np.argsort(fit)
        m_prev = m.copy()
        gen = 0
        
        while calls_made < budget:
            # Update state based on last generation
            x_sorted = pop[order]
            y_sorted = (x_sorted - m_prev) / sigma
            
            # Update mean
            m = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C @ y_sorted[:mu].T).sum(axis=1)
            
            # Update p_sigma
            pc_norm = np.linalg.norm(p_c)
            # h_sig calculation
            # We approximate the power term with gen
            h_sig = (pc_norm / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_sorted[:mu].T).sum(axis=1)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            # Rank-one and rank-mu updates
            # y_mean is the step of the mean
            y_mean = (m - m_prev) / sigma
            
            # Rank-one update
            C = (1 + c_1 * (h_sig * (2 - c_c) - 1)) * C + c_1 * np.outer(y_mean, y_mean)
            
            # Rank-mu update
            y_selected = y_sorted[:mu]
            C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->jk', y_selected, y_selected, y_selected), axis=0)
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update best
            idx_min = order[0]
            if fit[idx_min] < best_f:
                best_f = fit[idx_min]
                best_x = pop[idx_min].copy()
            
            # Prepare next generation
            if calls_made + lam > budget:
                # If we can't afford a full generation, we stop.
                # CMA-ES requires a full generation for proper adaptation.
                # We could do a smaller population but it breaks the assumptions.
                # Given budget=500 and lam=19 (for dim=5), 500/19 = 26 generations.
                # So we are fine.
                break
                
            m_prev = m.copy()
            pop = m + sigma * rng.multivariate_normal(np.zeros(dim), C, lam)
            pop = np.clip(pop, -5, 5)
            fit = np.array([func(x) for x in pop])
            calls_made += lam
            order = np.argsort(fit)
            gen += 1
            
        return best_x, best_f
