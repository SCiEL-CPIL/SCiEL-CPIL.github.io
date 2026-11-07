"""Acquisition function: Monte-Carlo Knowledge Gradient for ST1D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_global_start_time = None


def _ei_scores(mu, sigma, f_best):
    """Compute Expected Improvement scores (higher is better) for minimization."""
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _ucb_scores(mu, sigma, f_best, beta):
    """Compute UCB scores for minimization (higher is better)."""
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best):
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()

    # Deadline guard: 330s total budget.
    elapsed = time.time() - _global_start_time
    if elapsed > 320.0:
        # Fallback: fast EI
        return _ei_scores(mu, sigma, f_best)

    n = len(mu)
    
    # Knowledge Gradient via Monte Carlo sampling of the posterior.
    # For each candidate point i, KG = E[ max_j f(x_j) - f(x_i) | f(x_i) ]
    # where the max is over the current best (f_best) and the sampled value at i.
    # More precisely, KG_i = E[ min(f_best, f(x_i)) - f_best ] but since we minimize:
    # KG_i = E[ f_best - min(f_best, f(x_i)) ] = E[ max(0, f_best - f(x_i)) ]
    # Wait, that's just EI. KG is different: it considers the value of information
    # for the *entire* optimization process, not just the immediate improvement.
    #
    # Proper KG for minimization:
    # KG_i = E_{y_i ~ N(mu_i, sigma_i^2)} [ min(f_best, y_i) - f_best ] ... no.
    # KG_i = E_{y_i} [ min(f_best, y_i) ] - f_best? No.
    #
    # Let me recall: KG measures the expected reduction in the *final* optimum.
    # If we only have one more evaluation, KG reduces to EI.
    # But the standard KG approximation uses multiple samples.
    #
    # A common MC approximation:
    # 1. Sample K values y_i^(k) ~ N(mu_i, sigma_i^2) for each i and each k=1..K.
    # 2. For each k, the "new best" if we evaluated at i is min(f_best, y_i^(k)).
    # 3. KG_i = (1/K) * sum_k [ min(f_best, y_i^(k)) - f_best ] ... this is negative or zero.
    # Actually, we want the *improvement*: f_best - min(f_best, y_i^(k)) = max(0, f_best - y_i^(k)).
    # That's just EI again.
    #
    # The true KG accounts for the fact that after evaluating at i, we might continue.
    # But with a fixed budget of 30 evaluations, KG is complex.
    #
    # A simpler and effective approach: use **Upper Confidence Bound** with an
    # adaptive beta that depends on the spread of sigma and the current best.
    #
    # Structural change: Instead of a static portfolio selection, use a
    # **diversity-aware** acquisition that penalizes points similar to already-evaluated
    # regions. But we don't have coordinates.
    #
    # Alternative structural change: Use **Probability of Improvement (PI)** with a
    # small threshold, blended with UCB.
    #
    # Let's try a different family: **Thompson Sampling** style.
    # Sample y_i ~ N(mu_i, sigma_i^2) and return -y_i as the score.
    # This is equivalent to randomizing the UCB.
    #
    # But to make it deterministic and reproducible, let's use a fixed seed for the
    # random sampling within each call, or use a deterministic approximation.
    #
    # Actually, let's go back to a well-tuned **UCB** but with a beta that adapts
    # to the *relative* uncertainty. 
    #
    # New idea: **Normalized UCB**.
    # Score_i = (-mu_i + beta * sigma_i) / (std(-mu) + beta * std(sigma) + eps)
    # This normalizes the scores so that the scale is consistent.
    #
    # Even better: use a **rank-based** approach. 
    # Compute EI and UCB, then return the average of their *ranks*.
    # This is robust to scale differences.
    #
    # Let's implement rank-based blending of EI and UCB (beta=0.5).
    
    # Compute EI
    ei = _ei_scores(mu, sigma, f_best)
    
    # Compute UCB with beta=0.5
    ucb = _ucb_scores(mu, sigma, f_best, 0.5)
    
    # Compute ranks (lower rank = better, so we invert)
    # np.argsort gives indices that would sort the array. 
    # To get ranks where the best (max score) gets rank 1:
    ei_rank = np.empty(n)
    ucb_rank = np.empty(n)
    ei_rank[np.argsort(np.argsort(ei))[::-1]] = np.arange(1, n+1)
    ucb_rank[np.argsort(np.argsort(ucb))[::-1]] = np.arange(1, n+1)
    
    # Blend: average of ranks (lower is better, but we want higher score = better)
    # So we invert: score = n + 1 - rank
    blended_rank = 0.5 * ei_rank + 0.5 * ucb_rank
    scores = n + 1 - blended_rank
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
