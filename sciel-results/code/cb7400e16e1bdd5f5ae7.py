"""Thompson Sampling acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 325.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing a deterministic score like EI or UCB, we simulate
    a sample from the posterior distribution and return the negative of that sample.
    The evaluator selects the candidate with the highest score, which corresponds
    to the candidate with the lowest sampled value (minimization).
    
    To ensure exploration, we add a small amount of noise to the mean, scaled
    by the standard deviation. This is effectively a Thompson Sampling step.
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if time.time() > _DEALINE:
        # Pure exploitation fallback: minimize mu
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate a random sample from the normal distribution
    # We use a single random draw for all points to maintain correlation structure implicitly
    # Actually, for independent candidates, we can draw independent samples.
    # However, the standard Thompson Sampling in BO often involves sampling a function or using
    # the posterior at the points. Since we only have mu and sigma at the points,
    # we approximate by sampling from N(mu, sigma) for each point.
    
    # To avoid high variance in the selection, we can use a "noisy" mean:
    # sample = mu + z * sigma
    # We want to minimize this sample. So score = -sample = -mu - z * sigma.
    # But wait, the task says "higher meaning evaluate next".
    # If we want to pick the point with the lowest predicted value, we should return -predicted_value.
    # predicted_value ~ N(mu, sigma).
    
    # A common practical heuristic for Thompson Sampling in this context is:
    # score = mu + z * sigma
    # And then the loop picks the MAX score? No, for minimization, we want the MIN value.
    # So if we return scores where higher is better, and we want to pick the point with the lowest
    # sampled objective value, we should return:
    # score = - (mu + z * sigma)
    
    # Let's refine:
    # We draw z ~ N(0, 1) for each point.
    # sample = mu + z * sigma
    # We want to pick the point with the smallest sample.
    # Therefore, the score for "which one to pick" should be high for the smallest sample.
    # score = -sample = -mu - z * sigma
    
    # However, pure Thompson Sampling can be too exploitative if the GP is confident.
    # A slight modification often used is to add a small constant exploration term or
    # scale the sigma. Given the "Stagnation" directive and the need for a structural change,
    # let's stick to pure Thompson Sampling but ensure the random seed is handled correctly
    # or that it's robust.
    
    # One issue: if we just use -mu - z*sigma, the randomness might lead to picking a point
    # that is not the global minimum of the sample if the noise is too high.
    # But that is the point of exploration.
    
    # Let's use a fixed random state? No, we need it to be random for exploration.
    # But we need to ensure it's reproducible if needed? The task doesn't require reproducible
    # acquisition, just valid scores.
    
    # Let's try a hybrid: 
    # score = -mu + exploration_term
    # where exploration_term is derived from a sample.
    
    # Actually, the most direct interpretation of "Thompson Sampling" here is:
    # 1. Sample a value y_i ~ N(mu_i, sigma_i) for each candidate i.
    # 2. Select the candidate with the minimum y_i.
    # 3. To fit the interface "return 2000 finite scores, higher meaning evaluate next",
    #    we return -y_i. The max of -y_i corresponds to the min of y_i.
    
    # Implementation:
    # z = np.random.randn(len(mu))
    # y = mu + z * sigma_safe
    # scores = -y
    
    # Potential issue: If sigma is very small, this is just -mu.
    # If sigma is large, this explores.
    
    # Let's add a safety check for finite values.
    z = np.random.randn(len(mu))
    samples = mu + z * sigma_safe
    scores = -samples
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: If we want to bias towards exploitation more strongly, we could
    # scale the noise. But let's start with pure TS.
    
    return scores
# EVOLVE-BLOCK-END
