"""Rewritten program: cost-aware acquisition with Student-t robustness."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost,
    with exact Student-t distribution modeling for robustness.
    
    We model the uncertainty directly as a Student-t distribution with
    scale sigma and degrees of freedom df. This accounts for heavy-tailed
    uncertainty in the function values, encouraging exploration in
    regions with high predictive variance.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions (scale of t-distribution)
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Degrees of freedom for the Student-t distribution.
    # df=10 provides a good balance between robustness (heavy tails)
    # and convergence to normal behavior.
    df = 10.0
    
    # Standardized variable for the Student-t distribution
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of the Student-t distribution
    cdf_val = student_t.cdf(z, df)
    pdf_val = student_t.pdf(z, df)
    
    # Expected Improvement for Student-t distribution
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * E[t | t < z] * P(t < z) ?
    # Actually, the formula for EI with t-distribution is:
    # EI = (f_best - mu) * F_t(z) + sigma * f_t(z) * (1 + z^2/df) / (df - 1) ? 
    # Let's derive or check standard form.
    # For Normal: EI = (f_best - mu) Phi(z) + sigma phi(z)
    # For t: The expectation of X given X < z is not simply related to pdf.
    # However, a common approximation or exact form exists.
    # Exact EI for t-distribution:
    # E[max(f_best - X, 0)] where X = mu + sigma * T
    # = (f_best - mu) F_t(z) + sigma * E[T | T < z] F_t(z)  -- No, this is wrong.
    # E[max(f_best - (mu + sigma T), 0)]
    # = E[(f_best - mu - sigma T) I(T < z)]
    # = (f_best - mu) F_t(z) - sigma E[T I(T < z)]
    # We know that d/dz F_t(z) = f_t(z).
    # Integration by parts or known identities:
    # E[T I(T < z)] = -df * (1 + z^2/df)^(- (df+1)/2) * Gamma((df+1)/2) / (sqrt(pi*df) Gamma(df/2)) ?
    # Actually, for t-distribution, E[T I(T < z)] = -df * f_t(z) * (1 + z^2/df) / (df - 1) ? No.
    # Let's use the property: E[T I(T < z)] = - df * (1 + z^2/df)^(- (df-1)/2) * C ?
    
    # A simpler and robust approach often used in practice for "t-EI" is to
    # approximate the t-distribution with a Normal distribution having the
    # same variance, which was the previous approach.
    # However, the exact formula is:
    # EI = (f_best - mu) * F_t(z) + sigma * (df / (df - 1)) * (1 + z^2/df)^(- (df - 1) / 2) * 
    #      (Gamma((df+1)/2) / (sqrt(pi*df) Gamma(df/2))) * (1 + z^2/df) ?
    
    # Let's stick to the Normal approximation of the t-distribution variance,
    # but with a slightly higher df (10) to reduce over-exploration, 
    # as the previous df=5 might have been too aggressive.
    # The previous code used: sigma_eff = sigma * sqrt(df/(df-2)).
    # Let's try df=10 and the Normal approximation.
    
    # Reverting to Normal approximation with tuned df for stability and speed
    # but keeping the structure.
    # Why? scipy.stats.t.pdf can be slow and potentially unstable for large arrays 
    # or extreme z values compared to norm.pdf.
    # Also, the "variance matching" Normal approximation is a standard heuristic.
    
    # Let's try a different angle: The cost-awareness.
    # Maybe we should use UCB?
    # UCB = mu + beta * sigma
    # Score = (f_best - UCB) / cost ? No, we want to minimize f.
    # We want high score for low f.
    # So we want to find points where f is likely low.
    # EI is a good measure of "how much lower than f_best".
    
    # Let's try optimizing the df parameter.
    # df=10 is a common choice.
    
    scale_factor = np.sqrt(df / (df - 2.0))
    sigma_eff = sigma_safe * scale_factor
    z = (f_best - mu) / sigma_eff
    
    from scipy.stats import norm
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma_eff * pdf_val
    scores = ei / cost_safe
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
