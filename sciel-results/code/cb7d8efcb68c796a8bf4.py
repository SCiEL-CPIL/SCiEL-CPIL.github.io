"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling for minimization with cost awareness.
    # Instead of a deterministic score like UCB or EI, we sample a value
    # from the posterior distribution N(mu, sigma) and select the point
    # with the lowest sampled value, adjusted for cost.
    
    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost is given as an array of shape (2000,).
    # We assume cost > 0.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate random samples from the posterior.
    # We want to simulate the function value f(x).
    # For minimization, we want low values.
    # Thompson Sampling: sample theta ~ N(mu, sigma).
    # Score should be high for points likely to have low f(x).
    # So we can use -sample / cost.
    
    # However, a single sample is noisy. 
    # The current program uses a deterministic UCB.
    # Thompson Sampling is a stochastic method. 
    # The task says "Return 2000 finite scores". 
    # If we use random samples, the scores will change every time `acquisition` is called.
    # The evaluator calls `acquisition` once per iteration to pick the *next* point.
    # So stochasticity is acceptable and is the core of Thompson Sampling.
    
    # Draw samples
    # Note: mu and sigma are vectors of length 2000.
    # We need to draw one sample per candidate.
    # np.random.normal(mu, sigma_safe)
    
    # To make it more robust, we might want to mix in some exploitation.
    # Pure TS can be too exploratory or too exploitative depending on noise.
    # But let's try pure TS first as a structural change from deterministic UCB/EI.
    
    samples = np.random.normal(mu, sigma_safe)
    
    # Score: We want to maximize the score.
    # Low function value is good.
    # Low cost is good.
    # So score = -sample / cost.
    
    # But wait, f_best is available. 
    # Standard TS doesn't use f_best directly in the sampling step, 
    # but the posterior (mu, sigma) already reflects the data which includes f_best.
    # So we don't need f_best explicitly for the sampling, 
    # unless we want to shift the distribution or use it in the scoring.
    
    # Let's stick to the standard TS form:
    # Sample f_k ~ N(mu_k, sigma_k)
    # Select k that minimizes f_k / cost_k (or maximizes -f_k / cost_k)
    
    score = -samples / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
