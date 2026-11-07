# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a hybrid of Probability of Improvement (PI) and Expected Improvement (EI)
    to balance exploration and exploitation.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)

    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Hybrid acquisition: blend PI and EI
    # PI favors exploration (high probability of improvement regardless of magnitude)
    # EI favors exploitation (large expected improvement)
    # We use a weighted sum. The weight on PI can be tuned.
    # For 1D sphere, a moderate weight on PI helps escape local optima.
    # Let's try a simple blend: score = EI + alpha * PI
    # However, EI and PI have different scales. 
    # A better approach: use PI as the primary driver when EI is small, or use a normalized blend.
    # Alternatively, use a different acquisition function entirely, like UCB.
    
    # Let's try UCB (Upper Confidence Bound) for minimization:
    # UCB = mu - kappa * sigma
    # This encourages exploration by penalizing high uncertainty.
    # For minimization, we want to minimize mu - kappa * sigma.
    # So the score (higher is better) would be: -mu + kappa * sigma
    
    # Let's combine EI and UCB or just use a robust hybrid.
    # The directive suggests PI. Let's stick to a PI-based hybrid.
    # A common hybrid is to use PI to select candidates and EI to rank them, or a weighted sum.
    # Since we need a single score, let's try: score = pi * (1 + ei)
    # This scales the probability of improvement by the magnitude of improvement.
    
    # Actually, let's try a simpler structural change: Use PI directly, but with a twist.
    # Or use UCB. Let's try UCB with a kappa that decays or is fixed.
    # For 1D sphere, UCB with kappa=1 or 2 is often good.
    
    # Let's try a blend of EI and UCB.
    # UCB score (for minimization, higher is better): -mu + kappa * sigma
    # EI score: ei
    # We need to normalize them to comparable scales.
    
    # Let's try a different approach: Use the posterior mean and std to compute a score
    # that balances exploration and exploitation.
    
    # Let's try: score = -mu + beta * sigma
    # This is UCB. Let's set beta = 1.0.
    # But wait, the current program uses EI and scores 1.0.
    # Let's try PI as suggested.
    
    # PI = Phi(z)
    # This is between 0 and 1.
    # EI is unbounded.
    # If we return PI, we are purely exploring based on probability.
    # If we return EI, we are exploiting based on expected gain.
    
    # Let's try a hybrid: score = ei + pi
    # This adds a constant exploration bias.
    
    # Or, let's try: score = pi * exp(-ei) ... no, that's weird.
    
    # Let's try: score = pi / (sigma + eps)
    # This favors points with high probability of improvement and low uncertainty.
    
    # Let's try: score = pi * (1 + 1/sigma)
    # This favors high PI and low sigma.
    
    # Let's go back to the directive: "Use a Probability of Improvement (PI) acquisition function"
    # So let's try returning PI directly, or a modified PI.
    # PI = Phi(z)
    
    # Let's try: score = pi
    # This is a pure exploration strategy based on probability.
    
    # However, PI might be too exploratory. Let's blend it with exploitation.
    # score = pi * (1 + (f_best - mu) / (sigma + eps))
    # This is essentially EI + pi.
    
    # Let's try: score = ei + 0.1 * pi
    # A small exploration bonus.
    
    # Let's try: score = ei + pi
    # This gives equal weight to expected improvement and probability of improvement.
    
    # Let's compute both and return a blend.
    # We'll use: score = ei + pi
    
    score = ei + pi
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        return -mu
    
    return score
# EVOLVE-BLOCK-END
