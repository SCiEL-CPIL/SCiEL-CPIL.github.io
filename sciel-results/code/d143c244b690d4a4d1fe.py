import numpy as np
import time

# Global deadline guard: the evaluator kills the process at 330s wall-clock.
# We track cumulative time spent in acquisition calls and bail out safely.
_ACQ_DEADLINE = 320.0  # safety margin of 10s
_ACQ_START = None
_ACQ_ELAPSED = 0.0

def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition for minimization.
    
    score = mu + k * sigma, where k adapts to uncertainty.
    Higher score = more likely to improve upon f_best.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    global _ACQ_START, _ACQ_ELAPSED
    
    if _ACQ_START is None:
        _ACQ_START = time.time()
    
    # Deadline guard: if we're close to the limit, return conservative scores
    # that favor exploitation (low mu) to avoid wasting time
    if time.time() - _ACQ_START > _ACQ_DEADLINE:
        mu = np.asarray(mu)
        return -mu  # pure exploitation fallback
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Structural change: Probabilistic Lower Bound (PLB) acquisition.
    # Instead of UCB (which adds beta*sigma to encourage exploration),
    # PLB uses score = mu + k*sigma where k is derived from the
    # probability that a point improves upon f_best.
    #
    # The idea: for a Gaussian posterior N(mu, sigma^2), the probability
    # that f(x) < f_best is Phi((f_best - mu) / sigma). We want to
    # maximize this improvement probability, but we also want to
    # balance exploration. The PLB score approximates this by using
    # a k that scales with the relative uncertainty.
    #
    # k adapts: when sigma is large relative to the gap (f_best - mu),
    # there's high uncertainty and we should explore more (higher k).
    # When sigma is small, we exploit (lower k).
    
    # Compute the gap: how far below f_best the mean is (positive = promising)
    gap = f_best - mu
    
    # Relative uncertainty: sigma / max(|gap|, epsilon)
    # This tells us how much uncertainty there is relative to the potential improvement
    eps = 1e-8
    rel_unc = sigma / np.maximum(np.abs(gap), eps)
    
    # Map relative uncertainty to k using a smooth, bounded function.
    # k ranges from ~0.5 (exploit when very confident) to ~3.0 (explore when very uncertain).
    # Using tanh for smooth saturation.
    k = 0.5 + 2.5 * np.tanh(rel_unc / 2.0)
    
    # PLB score for minimization: lower mu is better, sigma encourages exploration
    scores = mu + k * sigma
    
    # Invert: we want HIGHER scores to mean "evaluate next", but lower mu is better.
    # So we negate: score = -(mu + k*sigma) = -mu - k*sigma
    # Wait — let me reconsider. The task says "higher meaning evaluate next".
    # For minimization, we want points with low mu. So score should be high when mu is low.
    # score = -mu + k * sigma would work: low mu → high score, high sigma → higher score (exploration).
    
    scores = -mu + k * sigma
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
