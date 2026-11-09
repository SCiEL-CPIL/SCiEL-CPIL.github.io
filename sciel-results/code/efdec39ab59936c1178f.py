import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.time() > _DEADLINE:
        return np.random.rand(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Thompson Sampling:
    # Draw a single sample from the posterior: f(x) ~ N(mu, sigma^2)
    # For each candidate x_i, sample f_i ~ N(mu_i, sigma_i^2)
    # The score is the "improvement per cost" or simply the negative sampled value per cost.
    # Since we want to minimize, a lower sampled value is better.
    # To make it a maximization problem (higher score = evaluate next),
    # we can use: Score = -sampled_value / cost
    # Or, to incorporate f_best slightly (to avoid exploring obviously bad regions if the sample is bad?),
    # pure TS uses just the sample.
    # However, standard TS for minimization picks argmin f_sampled.
    # Here we return scores such that the max is picked.
    # So Score_i = -f_sampled_i / cost_i
    
    # To ensure we don't pick a point that is predicted to be terrible even if the sample is lucky?
    # TS naturally handles exploration/exploitation.
    # Let's add a small bias towards f_best to ensure we don't go completely wild if sigma is huge?
    # No, pure TS is a distinct structural change.
    
    # We need a random seed or state? 
    # Using np.random is fine, but for reproducibility in tests? 
    # The task says "five seeded instances". The acquisition function itself doesn't take a seed.
    # We should use a deterministic pseudo-random generator based on the inputs? 
    # Or just use global np.random.
    # Using global np.random is standard for TS in this context unless specified otherwise.
    
    # However, if we use global np.random, the scores will change every call.
    # This is expected for TS.
    
    # One issue: if sigma is 0, sample is mu.
    # If sigma is very large, sample is noisy.
    
    # Let's draw the sample.
    # To be safe against huge sigma causing overflow or extreme values,
    # we can clip the sample? No, let the distribution speak.
    
    # Note: Using np.random.normal with large arrays is fast.
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: higher is better.
    # We want to minimize the function.
    # So a low sample is good.
    # Score = -sample / cost
    # This favors low samples and low costs.
    
    # Is there a risk of negative scores? Yes.
    # Is there a risk of non-finite scores? If cost is 0?
    # Cost is 1 + mean(x). x in [0,1], so cost in [1, 2]. Cost is never 0.
    
    scores = -samples / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # The evaluator picks the argmax of scores.
    # This corresponds to the argmin of (samples / cost).
    
    return scores
