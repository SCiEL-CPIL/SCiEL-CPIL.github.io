import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_SECONDS = 320.0
_eval_count = 0

def _elapsed():
    return time.time() - _start_time

def _ei(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def _ucb(mu, sigma, beta):
    return mu - beta * sigma

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard
    if _elapsed() > _DEADLINE_SECONDS:
        return _ei(mu, sigma, f_best)

    # Adaptive strategy
    # Early on: Explore more (UCB)
    # Later on: Exploit more (EI)
    
    # Total evals is 30.
    # If we are in the first 10, use UCB.
    # If we are in the last 20, use EI.
    # Or blend them.
    
    # Let's try a blend:
    # weight_ucb = max(0, 1 - _eval_count / 10) # 1.0 at start, 0.0 after 10
    # weight_ei = 1.0
    
    # But EI and UCB have different scales.
    # Better to stick to one or the other to avoid scale issues.
    
    # Let's try:
    # If _eval_count < 5: UCB with beta=2.0
    # If _eval_count < 15: UCB with beta=1.0
    # Else: EI
    
    # Actually, EI is often better for global minima if the GP is good.
    # The current program (score 2.0) uses a KG approx.
    # Let's try a **Pure EI** first to see if the KG approx was actually hurting or helping.
    # If the seed (score 1.0) was EI, then Pure EI will score 1.0.
    # If the seed was UCB, then Pure EI might be better.
    
    # The prompt says "seed program" scores 1.0.
    # The current program scores 2.0.
    # The current program is NOT pure EI. It's a KG approx.
    # So the KG approx improved the score.
    
    # To make a structural change, let's try **UCB** with a carefully chosen beta.
    # Or **PI** (Probability of Improvement).
    
    # Let's try **UCB** with a beta that depends on the current uncertainty.
    # beta = 1.0 + 0.5 * (f_best / (abs(f_best) + 1e-6)) ? No.
    
    # Let's try a **Mixture** of EI and UCB, normalized.
    # scores_ei = _ei(mu, sigma, f_best)
    # scores_ucb = _ucb(mu, sigma, 1.0)
    # Normalize both to [0, 1] range?
    # Then mix.
    
    # Let's try this:
    scores_ei = _ei(mu, sigma, f_best)
    
    # Add a small exploration term that is proportional to sigma
    # This is effectively UCB but added to EI?
    # EI = E[max(0, f_best - f(x))]
    # UCB = mu - beta * sigma
    
    # Let's try **EI** with a **shifted** f_best.
    # f_best_shifted = f_best - 0.1 * np.std(sigma)
    # This encourages exploring slightly worse regions.
    
    # Let's try **UCB** with a dynamic beta based on the number of evaluations.
    # Beta starts high (2.0) and decays to 0.5.
    beta = 2.0 * (1.0 - _eval_count / 30.0) + 0.5
    scores = _ucb(mu, sigma, beta)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
