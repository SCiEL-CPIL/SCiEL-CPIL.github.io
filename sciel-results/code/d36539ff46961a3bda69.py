import numpy as np

def acquisition(mu, sigma, f_best):
            # f_best is the current minimum observed value.
            # Initial f_best is likely around 5-25.
            # Target is 0.
            
            # Define beta based on f_best
            # If f_best is large, we are far from optimum -> explore more (high beta)
            # If f_best is small, we are close -> exploit more (low beta)
            
            # Normalize f_best to [0, 1] range roughly
            # f_best can be 0.
            # Let's say f_best > 5 is "far", f_best < 0.5 is "close"
            
            # Linear interpolation for beta
            # beta = 3.0 when f_best >= 5.0
            # beta = 0.5 when f_best <= 0.0
            
            # But we need it to be smooth?
            # Let's use a simple piecewise or clipped linear function.
            
            # beta = np.clip(0.5 + 0.5 * f_best, 0.5, 3.0)
            # If f_best = 0, beta = 0.5
            # If f_best = 5, beta = 3.0
            # If f_best = 25, beta = 3.0 (clipped)
            
            beta = np.clip(0.5 + 0.5 * f_best, 0.5, 3.0)
            
            # Calculate UCB score for minimization
            # We want to minimize mu - beta * sigma
            # So we maximize -(mu - beta * sigma)
            score = -mu + beta * sigma
            
            # Ensure finite
            score = np.where(np.isfinite(score), score, -np.inf)
            
            return score
