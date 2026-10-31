import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    M = len(bins)
    if M == 0:
        return np.array([])
        
    remaining = bins - item
    valid_mask = remaining >= 0
    
    scores = np.full(M, -np.inf)
    
    if not np.any(valid_mask):
        return scores
        
    # We only care about valid bins for the score, but the variance calculation
    # depends on the state of ALL bins.
    
    # Current mean
    mu = np.mean(bins)
    
    # New mean if we place item in bin i
    # mu_new = (sum(bins) - item) / M
    mu_new = mu - item / M
    
    # Base sum of squared deviations from mu_new for all bins
    # Note: We use bins (current remainders) for the "other" bins
    # and (bins - item) for the "chosen" bin
    deviations_base = bins - mu_new
    SS_base = np.sum(deviations_base ** 2)
    
    # For each bin i:
    # SS_i = SS_base - (bins[i] - mu_new)^2 + (bins[i] - item - mu_new)^2
    # Let delta = item
    # term_i = (bins[i] - mu_new)^2 - (bins[i] - item - mu_new)^2
    # SS_i = SS_base - term_i
    
    # We want to MINIMIZE variance, so we want to MINIMIZE SS_i.
    # Minimizing SS_i is equivalent to MAXIMIZING term_i.
    # So score can be term_i (higher is better).
    
    # term_i = A^2 - B^2 where A = bins[i] - mu_new, B = bins[i] - item - mu_new
    # A - B = item
    # A + B = 2*bins[i] - 2*mu_new - item
    # term_i = (A-B)(A+B) = item * (2*bins[i] - 2*mu_new - item)
    
    # So term_i is linear in bins[i]!
    # term_i = 2 * item * bins[i] - 2 * item * mu_new - item^2
    
    # If item > 0, term_i is maximized when bins[i] is MAXIMIZED.
    # This implies we should choose the bin with the LARGEST remaining capacity (Worst Fit)?
    # Let's re-verify.
    # Variance is minimized when the new value is close to the new mean.
    # If we put the item in a large bin, the new remainder is large - item.
    # If we put it in a small bin, the new remainder is small - item (close to 0).
    
    # Let's test with small numbers.
    # Bins: [10, 10]. Item: 1.
    # Case 1: Put in bin 0. New bins: [9, 10]. Mean 9.5. Var: ((9-9.5)^2 + (10-9.5)^2)/2 = 0.25.
    # Case 2: Put in bin 1. Symmetric.
    
    # Bins: [10, 5]. Item: 1.
    # Case 1: Put in 10. New: [9, 5]. Mean 7. Var: ((9-7)^2 + (5-7)^2)/2 = (4+4)/2 = 4.
    # Case 2: Put in 5. New: [10, 4]. Mean 7. Var: ((10-7)^2 + (4-7)^2)/2 = (9+9)/2 = 9.
    # Minimizing variance prefers Case 1 (Worst Fit).
    
    # Bins: [10, 1]. Item: 1.
    # Case 1: Put in 10. New: [9, 1]. Mean 5. Var: ((9-5)^2 + (1-5)^2)/2 = (16+16)/2 = 16.
    # Case 2: Put in 1. New: [10, 0]. Mean 5. Var: ((10-5)^2 + (0-5)^2)/2 = (25+25)/2 = 25.
    # Minimizing variance prefers Case 1 (Worst Fit).
    
    # It seems minimizing variance of remainders leads to Worst Fit behavior?
    # This is counter-intuitive to Best Fit being good.
    # However, the prompt says "minimize the variance ... thereby reducing the entropy of the gap distribution".
    # And "encourages a homogeneous set of gaps".
    # If we use Worst Fit, we keep gaps large and similar?
    # If we use Best Fit, we create small gaps and leave large gaps. High variance.
    
    # So, if the directive is to minimize variance, we should implement Worst Fit (or a variant).
    # But Best Fit (1.0) is the baseline.
    # The Weibull CDF score (1.031) was an improvement.
    # If Worst Fit is better, the score should be > 1.0.
    
    # Let's try a blend or just the Variance Minimization score.
    # Score = term_i = 2 * item * bins[i] - 2 * item * mu_new - item^2
    # Since we only care about the relative ranking for valid bins, and the constant terms don't depend on i,
    # we can just score by bins[i] (Worst Fit) if item > 0.
    
    # But wait, is it always Worst Fit?
    # If item is very large, say 9. Bins [10, 1].
    # Case 1: Put in 10. New [1, 1]. Mean 1. Var 0.
    # Case 2: Put in 1. Invalid.
    # So it picks the valid one.
    
    # What if Bins [10, 8]. Item 5.
    # Case 1: Put in 10. New [5, 8]. Mean 6.5. Var: ((5-6.5)^2 + (8-6.5)^2)/2 = (2.25+2.25)/2 = 2.25.
    # Case 2: Put in 8. New [10, 3]. Mean 6.5. Var: ((10-6.5)^2 + (3-6.5)^2)/2 = (12.25+12.25)/2 = 12.25.
    # Prefers Worst Fit.
    
    # So the "Variance Minimization" strategy is effectively **Worst Fit**.
    # Let's check if Worst Fit is a known good strategy.
    # Worst Fit is often worse than Best Fit for minimizing bins.
    # However, the prompt specifically directs this strategy.
    # "Strategy to try in this change: Model the packing process as a spectral clustering problem... minimize the variance...".
    
    # I will implement the Variance Minimization score.
    # score_i = 2 * item * bins[i] - 2 * item * mu_new - item^2
    # Since we want to maximize score (highest scoring bin), and item > 0,
    # this is equivalent to maximizing bins[i].
    # So it is Worst Fit.
    
    # But to be safe and follow the "smooth surrogate" hint, maybe I should add a small Best Fit component?
    # No, the directive is specific.
    
    # Let's just return the score derived from variance minimization.
    
    valid_indices = np.where(valid_mask)[0]
    
    # Calculate scores for valid bins
    # score = 2 * item * bins[valid_indices] - 2 * item * mu_new - item**2
    # The constant part - 2 * item * mu_new - item**2 is the same for all, so we can drop it for ranking.
    # So score = bins[valid_indices]
    
    # However, if we want to be precise and potentially allow for item < 0 (unlikely), we keep it.
    scores[valid_indices] = 2 * item * bins[valid_indices] - 2 * item * mu_new - item**2
    
    return scores
