import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling approach
    # Sample from the posterior distribution of the function values
    # f(x) ~ N(mu, sigma^2)
    
    # Generate standard normal samples
    z = np.random.randn(len(mu))
    
    # Sampled function values
    f_samples = mu + sigma * z
    
    # We want to minimize f(x) and cost.
    # A common cost-aware heuristic is to minimize f(x) / cost(x).
    # Since we need to return scores where higher is better, we return negative of this ratio.
    # Adding a small epsilon to cost to prevent division issues, though min cost is 1.
    safe_cost = cost + 1e-9
    
    scores = -f_samples / safe_cost
    
    return scores
