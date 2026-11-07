import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
       # Ensure sigma is not zero
       sigma = np.maximum(sigma, 1e-9)
       
       # Calculate EI
       z = (f_best - mu) / sigma
       ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
       
       # Adaptive exploration weight
       # If f_best is large, we are far from optimum, explore more.
       # If f_best is small, we are close, exploit more.
       # Scale f_best to [0, 1] roughly. Max sphere is 25.
       # Let's use a sigmoid or linear mapping.
       # weight = 0.5 + 0.5 * np.tanh(f_best / 5.0)
       # This gives weight in [0.5, 1.0] for f_best in [0, inf].
       # Wait, we want weight to be HIGH when f_best is HIGH.
       # tanh is positive and increasing.
       # So weight = 0.1 + 0.1 * np.tanh(f_best / 5.0)
       
       # Let's try:
       # lambda_exp = 0.1 * (1 + f_best)
       # This might be too aggressive.
       
       # Let's try a fixed small epsilon first, maybe that's enough?
       # The baseline EI scored 1.0.
       # Let's try EI with xi=0.1.
       # ei_xi = (f_best - mu - 0.1) * norm.cdf((f_best - mu - 0.1)/sigma) + sigma * norm.pdf((f_best - mu - 0.1)/sigma)
       
       # Let's try the hybrid EI + UCB idea from inspiration.
       # UCB_min = mu - beta * sigma
       # We want to maximize a score.
       # Score = EI + alpha * ( - (mu - beta * sigma) ) ? No.
       
       # Let's just use EI + epsilon * sigma.
       # epsilon = 0.05
       score = ei + 0.05 * sigma
       
       return score
