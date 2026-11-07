import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale function.
    Uses a Lower Confidence Bound (LCB) strategy with a fixed high exploration
    coefficient to effectively explore the flat basin of the Beale function.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Ensure sigma is positive and finite to avoid division or multiplication issues
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Lower Confidence Bound for minimization:
    # We want to minimize f(x). The LCB is mu(x) - beta * sigma(x).
    # To select the point with the lowest LCB, we maximize the score:
    # score(x) = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
    
    # Based on successful inspirations, a fixed beta of 5.0 provides a good
    # balance of exploration for this specific problem.
    beta = 5.0
    
    score = -mu + beta * sigma

    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, -1e10)

    return score
