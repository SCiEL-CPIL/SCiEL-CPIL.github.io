"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track evaluation count for adaptive parameters
_eval_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _eval_count
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    _eval_count += 1
    
    # Ensure finite outputs
    sigma = np.maximum(sigma, 1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # 1. Compute standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Higher EI means higher expected improvement (lower value).
    # Since we minimize, we want to maximize EI.
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # 2. Compute "Probability of Improvement" (PI)
    # PI = Phi(z)
    # This is 1 if mu < f_best (with high confidence), 0 otherwise.
    # It acts as a binary filter for exploitation.
    
    pi = norm.cdf(z)
    pi = np.clip(pi, 0.0, 1.0)
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 3. Structural Change: Hybrid Score with Adaptive Weighting
    # The current program uses UCB (mu - kappa*sigma), which is a linear combination.
    # EI is a non-linear function of mu and sigma that naturally balances exploration
    # and exploitation based on the distance to f_best.
    # 
    # However, pure EI can be too conservative in the early stages if f_best is far
    # from the true minimum, or too aggressive if it gets stuck in a local minimum.
    # 
    # We propose a "Gated EI" approach:
    # Score = EI * Gate
    # where Gate is a function that allows exploration when EI is low (uncertain)
    # and exploitation when EI is high (confident improvement).
    # 
    # Actually, a simpler and more robust structural change is to use a
    # "Weighted Sum" of EI and a pure exploration term (sigma), but with
    # adaptive weights based on the progress.
    # 
    # Let's try: Score = EI + lambda * sigma
    # This is equivalent to a modified EI that explicitly adds an exploration bonus.
    # 
    # But wait, EI already contains sigma * phi(z).
    # 
    # Let's look at the "Stagnation" signal. If the population best hasn't improved,
    # it might mean we are stuck. 
    # 
    # Alternative Structural Change: **Optimistic Initialization / Bootstrap Confidence**
    # Instead of a fixed formula, we can use a "Softmax" selection over a set of
    # acquisition functions? No, we must return a single score array.
    # 
    # Let's try **Expected Hypervolume Improvement (EHVI)** approximation for 1D?
    # No, too complex.
    # 
    # Let's go back to the difference between EI and UCB.
    # EI is myopic but theoretically grounded in expected gain.
    # UCB is myopic and grounded in confidence bounds.
    # 
    # A known issue with EI in BO is that it can be overly greedy.
    # A known issue with UCB is that the parameter kappa is hard to tune.
    # 
    # What if we use **PI (Probability of Improvement)** as a filter?
    # If PI is very low (mu > f_best + sigma), the point is unlikely to improve.
    # If PI is very high (mu < f_best - sigma), the point is likely to improve.
    # 
    # Let's try a **Piecewise** acquisition function:
    # If mu < f_best (likely improvement):
    #    Score = f_best - mu (Exploit: how much better is it?)
    # Else (uncertain or worse):
    #    Score = sigma (Explore: how uncertain is it?)
    # 
    # This is a hard switch. It might be too abrupt.
    # 
    # Let's use a **Smoothed** version:
    # Score = max(f_best - mu, 0) + sigma * w
    # 
    # Actually, let's look at the "Beale" function. It has a single global minimum at (3, 0.5) with value 0.
    # The domain is [-4.5, 4.5]^2.
    # 
    # The key to BO success is often finding the basin of attraction early.
    # EI is good for this.
    # 
    # Let's try a different structural change: **Variance-Weighted EI**.
    # Normalize EI by the scale of the problem?
    # 
    # Let's try **UCB with a dynamic kappa derived from the data**.
    # kappa = median(sigma) / median(f_best - mu) ? No.
    # 
    # Let's stick to a robust, standard method but implemented differently.
    # **Expected Improvement with a noise parameter xi that adapts to the range of f.**
    # 
    # f_range = max(mu) - min(mu)
    # xi = 0.05 * f_range
    # 
    # This makes the acquisition function scale-invariant.
    # If the function values are large, xi is large (more exploration).
    # If the function values are small (converging), xi is small (more exploitation).
    # 
    # This is a structural change: **Scale-Adaptive EI**.
    
    # Compute a scale estimate from the current posterior means
    # Using the interquartile range (IQR) is robust to outliers
    q75, q25 = np.percentile(mu, [75, 25])
    scale = q75 - q25
    if scale < 1e-6:
        scale = 1.0
    
    # Adaptive xi: 5% of the scale
    xi = 0.05 * scale
    
    # Compute EI with xi
    # z = (f_best - xi - mu) / sigma
    z_xi = (f_best - xi - mu) / sigma
    ei_xi = (f_best - xi - mu) * norm.cdf(z_xi) + sigma * norm.pdf(z_xi)
    
    # Ensure non-negative
    ei_xi = np.maximum(ei_xi, 0.0)
    ei_xi = np.nan_to_num(ei_xi, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Return EI_xi as the score
    return ei_xi

# EVOLVE-BLOCK-END
