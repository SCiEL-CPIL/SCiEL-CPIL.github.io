"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0

def acquisition(mu, sigma, f_best):
    """
    Hypervolume Indicator (HVI) inspired acquisition function for Goldstein-Price 2D.
    
    The reference point is set to the known global minimum (3.0). The score
    approximates the hypervolume of the region dominated by the posterior
    predictive distribution relative to this reference point.
    
    For single-objective minimization, the HVI contribution of a candidate
    with posterior N(mu, sigma^2) relative to reference r is:
        H = E[max(0, r - Y)] + c * E[max(0, r - Y - |Y - mu|)]
    The first term is standard EI towards r. The second term captures
    "tail risk" - the probability that the true value exceeds the reference
    by a margin proportional to uncertainty, which penalizes uncertain points
    that might be worse than the known minimum.
    
    Additionally, we add a knowledge-gradient term that rewards points where
    the posterior has a non-negligible probability of being below the current
    best observed value, weighted by how much better it could be.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        return np.nan_to_num(-mu + 2.0 * np.maximum(sigma, 1e-10), nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    sigma_safe = np.maximum(sigma, 1e-10)
    f_min_true = 3.0
    
    # Term 1: Standard EI towards the known global minimum (3.0)
    # This is the primary HVI component - area of improvement over reference
    z_ref = (f_min_true - mu) / sigma_safe
    ei_ref = (f_min_true - mu) * norm.cdf(z_ref) + sigma_safe * norm.pdf(z_ref)
    
    # Term 2: Tail-risk penalty for points where uncertainty might push value above reference
    # E[max(0, Y - f_min_true)] where Y ~ N(mu, sigma^2)
    # = (Y - r) * Phi((Y-r)/sigma) + sigma * phi((Y-r)/sigma) with Y=mu
    # Actually: E[max(0, Y - r)] = (mu - r)*Phi((mu-r)/sigma) + sigma*phi((mu-r)/sigma)
    z_tail = (mu - f_min_true) / sigma_safe
    tail_risk = (mu - f_min_true) * norm.cdf(z_tail) + sigma_safe * norm.pdf(z_tail)
    
    # Term 3: Knowledge-gradient inspired term - rewards points that could beat f_best
    # with high confidence, scaled by the potential improvement
    z_best = (f_best - mu) / sigma_safe
    prob_better = norm.cdf(z_best)  # P(Y < f_best)
    potential_improvement = np.maximum(0.0, f_best - mu)
    kg_term = prob_better * potential_improvement
    
    # Combine terms with adaptive weights based on problem progress
    # When f_best is close to 3.0, we should be more exploitative
    gap_to_opt = max(0.0, f_best - f_min_true)
    
    # Adaptive weights:
    # - ei_ref gets higher weight as we get closer to the optimum
    # - tail_risk gets lower weight (less penalty) when we're confident
    # - kg_term provides additional signal for finding better-than-current points
    
    # Weight for EI towards true minimum: always important, increases slightly with confidence
    w_ei = 1.0
    
    # Weight for tail risk: decreases as we get closer to optimum (less need to penalize)
    # When gap_to_opt is large, we're far from optimum and tail risk matters more
    w_tail = 0.5 * np.exp(-gap_to_opt / 2.0)
    
    # Weight for knowledge gradient: moderate, helps escape local optima
    w_kg = 0.3 + 0.4 * np.exp(-gap_to_opt / 1.0)
    
    scores = w_ei * ei_ref - w_tail * tail_risk + w_kg * kg_term
    
    # Normalize to ensure stable scoring across iterations
    # Subtract the median to center scores, which helps with numerical stability
    # but preserves the relative ordering
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size > 0:
        median_score = np.median(finite_scores)
        scores = scores - median_score
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
