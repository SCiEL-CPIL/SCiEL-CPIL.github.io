import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses a modified Probability of Improvement (POI) strategy:
    Score = P(Improvement) * Expected_Gain
    This balances the likelihood of finding a better point with the magnitude
    of the improvement, distinct from pure EI or UCB.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin

    # Input validation and sanitization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and add small epsilon for numerical stability
    eps = 1e-8
    sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu.
    z = (f_best - mu) / sigma
    
    # Compute CDF (Probability of Improvement) and PDF
    # P(f(x) < f_best) = Phi(z)
    # phi(z) is the PDF, used in the expectation calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Gain (same as the term inside EI's expectation, but without the probability weight)
    # E[improvement | improvement] is not quite right, let's look at the components.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # POI = Phi(z)
    # The directive suggests: "Probability of Improvement ... scaled by the magnitude of the expected gain"
    # Let's define Expected Gain as the expected improvement value, which is EI itself.
    # However, a common variant is simply POI * (f_best - mu) or similar.
    # Let's stick to the interpretation: Score = POI * (Expected Improvement)
    # But EI is already (f_best - mu)*Phi(z) + sigma*phi(z).
    # If we use Score = Phi(z) * ((f_best - mu) * Phi(z) + sigma * phi(z)), it becomes a squared probability term.
    
    # Alternative interpretation: "scaled by the magnitude of the expected gain"
    # Gain = f_best - mu (deterministic mean gain)
    # Score = POI * (f_best - mu)
    # This is a simpler, more distinct structural change than EI.
    # EI is roughly (f_best - mu) * P(imp) + uncertainty_term.
    # POI * (f_best - mu) removes the uncertainty term (sigma * phi(z)) and multiplies the mean gain by the probability.
    # This strongly favors high-probability improvements, potentially more than EI does in high-uncertainty regions.
    
    expected_gain = f_best - mu
    poi = cdf_z
    
    # The score is the product of the probability of improvement and the expected gain.
    # Note: If mu > f_best, expected_gain is negative, but POI is small. 
    # To ensure we only consider positive gains, we can clamp expected_gain to 0?
    # No, EI handles this naturally. If mu > f_best, z < 0, POI < 0.5.
    # Actually, if mu > f_best, the "gain" is negative, so we shouldn't pick it.
    # EI yields small positive values for mu > f_best due to the sigma*phi term.
    # POI * (f_best - mu) would be negative if mu > f_best.
    # We want higher scores to be better.
    # So, if mu > f_best, the score should be low (or zero).
    # Let's use np.maximum(f_best - mu, 0) to ensure we only reward actual potential improvements.
    
    safe_gain = np.maximum(f_best - mu, 0.0)
    
    scores = poi * safe_gain
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() > deadline:
         pass

    return scores
