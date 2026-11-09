import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    zero = np.zeros(n, dtype=np.float64)
    
    # Safety: if inputs are degenerate, return a neutral score.
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return zero
    
    # Handle zero sigma and cost to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # ---- Expected Improvement (Minimization) ----
    # imp = f_best - mu (positive if mu < f_best)
    imp = f_best - mu
    s = sigma_safe
    z = imp / s
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = imp * cdf(z) + sigma * pdf(z)
    ei = imp * cdf_z + s * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # ---- Probability of Improvement (PoI) ----
    # PoI = cdf(z)
    poi = cdf_z
    
    # ---- Worst-Case Improvement (WCI) / LCB on improvement ----
    # WCI = max(0, imp - k * sigma)
    # We use a fixed k=1.0 for robust minimax regret penalty.
    # This penalizes high uncertainty more aggressively than EI in some regimes.
    k = 1.0
    wci = imp - k * s
    wci = np.maximum(wci, 0.0)
    
    # ---- Cost-aware Scores ----
    # We want to maximize the score.
    # Score 1: EI / cost
    score_ei = ei / cost_safe
    
    # Score 2: WCI / cost
    score_wci = wci / cost_safe
    
    # Score 3: Blended EI with PoI confidence (from inspiration 1)
    # This amplifies points where improvement is both expected and likely.
    blended_ei = ei * (1.0 + poi)
    score_blended = blended_ei / cost_safe
    
    # ---- Ensemble Strategy ----
    # We combine these scores. 
    # Using max() picks the most aggressive recommendation.
    # Using a weighted sum allows for smoother blending.
    # Given the directive to try WCI and combine with current (EI-based),
    # and inspiration 1 (EI*PoI), we will use a max-based ensemble to ensure
    # we don't miss a strong signal from any single metric, but we can also
    # normalize to prevent scale issues. However, simple max is often robust.
    
    # Let's try a hybrid: primarily the blended EI (which showed promise), 
    # but boosted by WCI if it's significantly higher, or just max of the two.
    # Actually, let's look at the scales. EI and WCI are in the same units.
    # Blended EI is roughly 2x EI at best.
    
    # Let's use max(score_ei, score_wci) as the base, then multiply by a confidence factor?
    # Or simply max(score_blended, score_wci)?
    
    # Let's stick to a clear structural change:
    # Primary: Blended EI (EI * (1+PoI)) / cost
    # Secondary: WCI / cost
    # We take the element-wise maximum to allow either the probabilistic expected gain
    # or the robust worst-case gain to drive the selection.
    
    scores = np.maximum(score_blended, score_wci)
    
    # Ensure finite, non-negative.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Deadline guard: return the current (already computed) result safely.
    if time.time() - start_time > 325:
        return scores
        
    return scores
