# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Monte Carlo Knowledge Gradient (KG) with adaptive exploration.
    
    The Knowledge Gradient maximizes the expected reduction in the best observed value:
    KG(x) = f_best - E[min(Y, f_best)] where Y ~ N(mu(x), sigma(x)^2).
    
    This is computed via Monte Carlo sampling. We add a small UCB-style exploration term
    to avoid premature convergence in the early stages when f_best is far from the optimum.
    
    The exploration weight decays as the maximum posterior uncertainty decreases,
    shifting the balance from exploration to exploitation.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    n_candidates = len(mu)
    
    # Monte Carlo samples for KG
    # 200 samples provides a good balance between accuracy and speed
    n_samples = 200
    
    # Generate standard normal samples: shape (n_samples, n_candidates)
    z = np.random.randn(n_samples, n_candidates)
    
    # Transform to posterior samples
    Y = mu[None, :] + sigma[None, :] * z
    
    # Compute min(Y, f_best) for each sample and candidate
    min_Y = np.minimum(Y, f_best)
    
    # Expected minimum
    E_min_Y = np.mean(min_Y, axis=0)
    
    # Knowledge Gradient: expected reduction in best value
    kg = f_best - E_min_Y
    
    # Time check after MC computation
    elapsed = time.time() - start_time
    
    # Adaptive exploration term
    # Scale UCB by a factor that decreases as we get closer to the optimum
    max_sigma = np.max(sigma)
    
    # Exploration weight: high when max_sigma is high (early), low when it's low (late)
    # max_sigma is in objective units (standardized), typically starts around 1.0
    weight_explore = np.clip(max_sigma, 0.0, 1.0) * 0.8
    
    # UCB exploration term: -beta * sigma (we want high scores, and lower f is better)
    # Use a fixed beta scaled by the exploration weight
    beta = 0.5
    ucb = -beta * sigma
    
    # Combined score: KG + exploration
    scores = kg + weight_explore * ucb
    
    # Time check: if we're getting close to the deadline, return a simpler score
    if time.time() - start_time > 310:
        # Fallback: just use KG without exploration (or -mu for pure exploitation)
        scores = kg
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
