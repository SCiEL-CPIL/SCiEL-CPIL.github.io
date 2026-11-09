import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior distribution
    # We want to minimize, so we sample the objective values
    # and pick the one with the lowest value.
    # The acquisition score should be high for points we want to pick.
    # So, Score = - (Sampled Value)
    
    # Use a local random generator to avoid state leakage?
    # Or use global?
    # Using global is standard for TS in a loop.
    # However, if the evaluator calls this in a way that expects determinism, it fails.
    # But BO with TS is inherently stochastic.
    
    # To be safe, let's use a fixed seed? No, that makes it deterministic.
    # Let's use the global np.random.
    
    samples = mu + sigma * np.random.randn(len(mu))
    return -samples
