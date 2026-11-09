import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
       sigma = np.maximum(sigma, 1e-12)
       cost = np.maximum(cost, 1e-12)
       
       z = (f_best - mu) / sigma
       ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
       
       # Add a pure exploration term to encourage visiting unknown areas
       # This is structural: we are not just scaling EI, we are adding a distinct exploration component
       # and then normalizing by cost.
       
       # Beta for exploration
       # High beta to ensure we don't get stuck
       beta = 1.5
       
       # Exploration component: f_best - mu + beta * sigma
       # This is the lower confidence bound (for minimization)
       # We want to maximize this (make it less negative / more positive relative to f_best)
       # Actually, for minimization, we want low values.
       # UCB for minimization: L(x) = mu(x) - beta * sigma(x).
       # We want to minimize L(x).
       # So we want to maximize -L(x) = -mu(x) + beta * sigma(x).
       # But we also want improvement over f_best.
       
       # Let's use the standard EI which already balances this.
       # But EI goes to 0 when mu >> f_best.
       # The exploration term prevents this.
       
       exploration = -mu + beta * sigma
       
       # Hybrid
       # Weight 0.5 EI, 0.5 Exploration
       hybrid = 0.5 * ei + 0.5 * exploration
       
       # Normalize by cost
       scores = hybrid / cost
       
       scores = np.where(np.isfinite(scores), scores, 0.0)
       
       return scores
