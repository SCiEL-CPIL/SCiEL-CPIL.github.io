import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0

def acquisition(mu, sigma, f_best):
        if time.time() > _DEADLINE:
            return -mu + 1.0 * sigma

        sigma_safe = np.maximum(sigma, 1e-9)
        
        f_min_global = -3.32237
        f_initial_guess = 1.0
        
        f_best_clamped = np.clip(f_best, f_min_global, f_initial_guess)
        t = (f_best_clamped - f_initial_guess) / (f_min_global - f_initial_guess)
        t = np.clip(t, 0.0, 1.0)
        
        # Adaptive Beta for Stochastic UCB
        beta_mean = 2.5 - 2.0 * t  # 2.5 -> 0.5
        beta_std = 0.5 - 0.4 * t   # 0.5 -> 0.1
        
        # Draw beta for this step
        beta = np.random.normal(beta_mean, beta_std)
        # Ensure beta is positive to avoid flipping the objective
        beta = max(beta, 0.1)
        
        scores = -mu + beta * sigma_safe
        
        # Optional: Add a small EI component for stability in late stage
        # If t > 0.8, blend with EI
        if t > 0.8:
            z = (f_best - mu) / sigma_safe
            from scipy.stats import norm
            ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            
            # Normalize EI
            ei_min = np.min(ei)
            ei_max = np.max(ei)
            if ei_max > ei_min:
                ei_norm = (ei - ei_min) / (ei_max - ei_min)
            else:
                ei_norm = np.zeros_like(ei)
                
            # Normalize UCB
            ucb_min = np.min(scores)
            ucb_max = np.max(scores)
            if ucb_max > ucb_min:
                ucb_norm = (scores - ucb_min) / (ucb_max - ucb_min)
            else:
                ucb_norm = np.zeros_like(scores)
                
            # Blend
            w_ei = (t - 0.8) / 0.2 * 0.5 # Max 0.5 weight for EI
            scores = (1 - w_ei) * ucb_norm + w_ei * ei_norm
            
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores
