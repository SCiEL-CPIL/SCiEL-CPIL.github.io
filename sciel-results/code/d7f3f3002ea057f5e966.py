import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Expected Improvement (Minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Worst-Case Improvement (WCI) for minimization
    # Pessimistic estimate of the function value: mu + k * sigma
    # We only consider it an improvement if this pessimistic value is better than f_best
    # Score = max(0, f_best - (mu + k * sigma))
    k = 1.0  # Fixed exploration constant
    
    wci = f_best - mu - k * sigma_safe
    wci = np.maximum(wci, 0.0)
    
    # Adaptive blend:
    # When uncertainty (sigma) is high, the GP is less confident, so we rely more on
    # the pessimistic WCI to avoid being misled by high-variance predictions.
    # When uncertainty is low, the GP is confident, so we rely on the expected EI.
    #
    # We normalize sigma to a [0, 1] weight using a logistic-like scaling or simple clamp.
    # Let's use a simple heuristic: if sigma is large relative to the scale of f_best-mu,
    # we trust WCI more.
    # A robust way is to look at the z-score.
    # If z is very positive (mu << f_best), EI is large and WCI is large.
    # If z is very negative (mu >> f_best), EI is ~0 and WCI is 0.
    # If z is around 0, there is high uncertainty.
    
    # Let's blend based on sigma magnitude relative to a scale.
    # However, sigma is in objective units.
    # A simpler structural change:
    # Use a weighted sum where the weight for WCI increases with sigma.
    
    # Let's try a different structural idea from the inspirations:
    # The "Adaptive PoI/EI blend" mentioned in the cell findings.
    # PoI (Probability of Improvement) = Phi(z)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Let's combine EI and WCI.
    # WCI is a "safe" improvement. EI is an "expected" improvement.
    # In multimodal functions, EI can be attracted to high-variance regions that might
    # not actually contain the global minimum (over-exploration).
    # WCI is more conservative.
    
    # Let's use a blend:
    # score = alpha * EI + (1 - alpha) * WCI
    # where alpha depends on confidence.
    
    # Confidence can be proxied by 1/sigma.
    # If sigma is small, alpha should be high (trust EI).
    # If sigma is large, alpha should be low (trust WCI, or rather, be more conservative).
    
    # However, WCI is 0 when mu + k*sigma > f_best.
    # EI is always >= 0.
    
    # Let's try a simpler combination that has shown success:
    # The inspiration 1 (EI/cost) scored 1.0.
    # The current program (WCI/cost) scored 0.90.
    # The cell finding mentions "Adaptive PoI/EI blend ... improved to 1.236".
    # Let's implement an adaptive blend of EI and a PoI-like term, or EI and WCI.
    
    # Let's look at the "Adaptive PoI/EI blend" more closely.
    # PoI = norm.cdf(z)
    # EI = (f_best - mu) * PoI + sigma * norm.pdf(z)
    
    # Perhaps the "blend" refers to blending the *acquisition function* based on the
    # number of evaluations or the current best value?
    # But we don't have access to the iteration count directly, only f_best.
    
    # Let's try blending EI and WCI based on the magnitude of sigma.
    # We need a scale for sigma. Since targets are standardized, sigma is in objective units.
    # The range of f is roughly [-10.5, 10].
    # Let's assume a characteristic scale of sigma is around 1.0.
    
    # Let's use a soft switch.
    # weight_wci = 1 / (1 + exp(-sigma))  # sigmoid of sigma
    # This makes weight_wci ~ 0.5 when sigma ~ 0, and approaches 1 as sigma increases.
    # Wait, if sigma is large, we want to be more conservative (WCI).
    # If sigma is small, we want to be more aggressive (EI).
    
    # So:
    # weight_ei = 1 / (1 + exp(sigma))   # decreases with sigma
    # weight_wci = 1 - weight_ei         # increases with sigma
    
    # Let's refine the scale.
    # If sigma is 0, weight_ei = 0.5, weight_wci = 0.5.
    # If sigma is 5, weight_ei ~ 0.006, weight_wci ~ 0.994.
    # If sigma is -5 (impossible), weight_ei ~ 0.994.
    
    # This seems reasonable.
    
    # However, WCI is 0 in many cases where EI is non-zero.
    # If WCI is 0, the score becomes 0, even if EI is high.
    # This might be too conservative.
    
    # Alternative: Use EI as the base, and penalize high uncertainty if the improvement is not "guaranteed".
    # Or, use the "Probability of Improvement" (PoI) which is smoother than WCI.
    
    # Let's try a blend of EI and PoI.
    # PoI = norm.cdf(z)
    # PoI is 1 if mu << f_best, 0 if mu >> f_best.
    # EI includes the magnitude of improvement.
    
    # The cell finding says: "Adaptive PoI/EI blend with uncertainty-scaled exploration bonus".
    # This suggests:
    # Score = blend(PoI, EI) + exploration_bonus * sigma
    
    # Let's implement this.
    # Blend PoI and EI.
    # PoI is dimensionless [0,1]. EI is in objective units.
    # We can't directly add them without scaling.
    
    # Let's stick to the EI/cost which scored 1.0, and add a structural modification
    # to handle the "stagnation" or "multimodal" issue.
    
    # The current WCI approach failed (0.90).
    # The EI approach worked (1.0).
    # The "Adaptive PoI/EI blend" worked better (1.236).
    
    # Let's reconstruct the "Adaptive PoI/EI blend".
    # It likely uses PoI when uncertainty is high to avoid being fooled by high-variance
    # peaks that don't actually improve f_best, and EI when uncertainty is low to exploit
    # known improvements.
    
    # But PoI doesn't account for the *size* of the improvement, only the probability.
    # A small improvement with high probability is preferred over a large improvement with low probability?
    # Usually EI is better because it weighs the magnitude.
    
    # Maybe the blend is:
    # If sigma > threshold, use PoI (conservative, probability-based).
    # If sigma < threshold, use EI (aggressive, magnitude-based).
    
    # Let's try a soft switch again.
    # score = alpha * EI + (1 - alpha) * PoI * scale_factor
    
    # What is a good scale_factor for PoI to match EI?
    # EI is roughly (f_best - mu) when mu << f_best.
    # PoI is 1 when mu << f_best.
    # So if we multiply PoI by (f_best - mu), we get something similar to EI?
    # Actually, EI = (f_best - mu) * PoI + sigma * pdf.
    # The second term is the "uncertainty bonus".
    
    # So EI is already a blend of "expected improvement" and "uncertainty bonus".
    
    # Why would PoI be better?
    # PoI = Phi(z).
    # If z is large, PoI ~ 1. EI ~ f_best - mu.
    # If z is small, PoI ~ 0.5. EI ~ sigma * phi(0) = sigma * 0.4.
    
    # Perhaps the "blend" refers to blending EI with a term that penalizes high cost more heavily?
    # We already divide by cost.
    
    # Let's look at the "uncertainty-scaled exploration bonus".
    # Maybe the score is:
    # EI / cost + lambda * sigma / cost ?
    # This is just EI + lambda*sigma, which is LCB-like.
    
    # Let's try a different structural change:
    # Instead of a fixed EI or WCI, use a **Thompson Sampling** style approach?
    # No, we only have mu and sigma.
    
    # Let's go back to the "Adaptive PoI/EI blend".
    # Maybe it blends EI and **Probability of Improvement** to create a more robust signal.
    # Let's try:
    # score = (EI + PoI * (f_best - mu)) / cost
    # This doubles down on the improvement magnitude when probability is high.
    
    # Or maybe:
    # score = EI / cost * (1 + PoI)
    # This boosts the score when the probability of improvement is high.
    
    # Let's try this:
    # Base score is EI / cost.
    # Multiply by a factor that increases with PoI.
    # Factor = 1 + PoI.
    # If PoI is 0 (no chance of improvement), score is EI (which is also 0).
    # If PoI is 1 (certain improvement), score is 2 * EI.
    # This effectively weights the expected improvement by the confidence in it.
    
    # Let's test this hypothesis.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # PoI = Phi(z)
    
    # score = EI * (1 + PoI) / cost
    
    # This is a simple structural change to the EI formula.
    
    # Let's implement this.
    
    poi = cdf_z  # Probability of Improvement
    
    # Ensure PoI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Blend factor
    blend_factor = 1.0 + poi
    
    scores = ei * blend_factor / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
