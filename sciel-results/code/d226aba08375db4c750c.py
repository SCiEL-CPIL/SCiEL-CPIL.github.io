"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (minimization perspective)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Hybrid strategy: Combine EI with a UCB-like exploration term.
    # The current program (EI/cost) scores 1.0. 
    # Inspiration 1 (UCB/cost) scored lower (0.73), suggesting pure UCB is too aggressive or poorly scaled.
    # Inspiration 2 (EI + lambda*sigma)/cost scored lower (0.61), suggesting the specific lambda or interaction was suboptimal.
    
    # We will stick to the core structure of the current best program (EI/cost) 
    # but add a very small, adaptive exploration bonus to ensure we don't get stuck 
    # in local optima if sigma is non-negligible, while keeping the cost-awareness.
    
    # A small constant beta for exploration in the numerator.
    # Since EI is in objective units, and sigma is in objective units, we can add a small multiple of sigma.
    # However, the current program is already at the seed level (1.0). 
    # Often, pure EI is a strong baseline. 
    # Let's try a slight modification: use a slightly more aggressive exploitation/exploitation balance 
    # by ensuring that if EI is very low (near 0), we still have a small chance to explore based on sigma, 
    # but normalized by cost.
    
    # Actually, looking at the inspirations, they both underperformed the seed.
    # The seed is `EI / cost`.
    # Let's try to refine the seed by handling the case where sigma is very small but mu is good.
    # The seed already does this.
    
    # One potential improvement: The cost function is `1 + mean(x)`. 
    # This implies points closer to the origin are cheaper.
    # The seed divides by cost, effectively prioritizing cheap points with high EI.
    
    # Let's try a "Weighted" approach where we blend EI and a local uncertainty measure, 
    # but keep the weighting very close to the seed to avoid the pitfalls of the inspirations.
    
    # Let's try: Score = (EI + epsilon * sigma) / cost
    # If epsilon is very small, it's close to the seed.
    # If epsilon is large, it becomes UCB-like.
    # Given Inspiration 1 (pure UCB) failed, and Inspiration 2 (EI + 0.1*sigma) failed,
    # it's possible that the seed is already near-optimal for this specific setup, 
    # or that the "exploration" term is hurting because the GP is well-calibrated.
    
    # Alternative: Use a slightly different normalization for EI to make it more robust?
    # Or perhaps the issue is that `cost` varies significantly.
    
    # Let's try a different angle: The "Hartmann 3D" is multimodal.
    # EI is good for exploitation. UCB is good for exploration.
    # In early stages (low budget), exploration is key.
    # The loop runs while total cost < 30.
    # Initial 5 points cost ~ 5 * 1.5 = 7.5.
    # Remaining budget ~ 22.5.
    # Each eval costs ~ 1.5. So ~ 15 more evals.
    # Total ~ 20 evals.
    
    # With only 20 evals, we need to be very careful.
    # The seed program is likely strong.
    # Let's try to combine the *structural* idea of cost-awareness (from seed) 
    # with a *refinement* that adapts the exploration rate based on the remaining budget?
    # But we don't have access to the total budget or step count in the function signature.
    
    # Let's look at the inputs: mu, sigma, f_best, cost.
    # We can estimate the "stage" of optimization by looking at f_best.
    # If f_best is close to the known minimum (-3.86278), we should exploit more.
    # If f_best is far, we should explore more.
    
    # Let's define a dynamic beta based on how good f_best is.
    # However, we don't know the exact minimum in the function signature, 
    # but we know it's around -3.86 for Hartmann 3D.
    
    # Let's try a simple adaptive beta:
    # If f_best is very good (e.g., < -3.0), use small beta.
    # If f_best is poor (e.g., > -2.0), use larger beta.
    
    # But wait, the inspirations used fixed betas and failed.
    # Maybe the failure was due to the specific form `(beta*sigma - mu) / cost` (UCB) vs `(EI + beta*sigma)/cost`.
    # The seed is `(EI) / cost`.
    
    # Let's try to keep the seed's form but add a tiny epsilon to prevent division by zero 
    # if cost is extremely small (though cost >= 1).
    
    # Actually, let's try a different combination:
    # Use EI, but scale it by a factor that encourages exploration when sigma is high relative to the gap (f_best - mu).
    
    # Let's try: Score = EI / cost * (1 + alpha * sigma / (f_best - mu + epsilon))
    # This amplifies the score for points with high uncertainty relative to their potential improvement.
    
    # Let's keep it simple and robust. The seed is 1.0.
    # Let's try a very slight modification to the EI calculation to handle numerical stability better 
    # or to slightly favor exploration in the early stages.
    
    # Let's try: Score = (EI + 0.01 * sigma) / cost
    # This is very close to the seed.
    
    # Alternatively, let's try the "Upper Confidence Bound" but blended with EI more softly.
    # Score = (0.9 * EI + 0.1 * (beta * sigma - mu)) / cost
    # This is a blend.
    
    # Let's go with a blend that is heavily weighted towards EI (the seed's strength) 
    # but includes a small UCB component to help escape local minima.
    
    beta = 1.0
    ucb_term = beta * sigma_safe - mu
    ei_term = ei
    
    # Blend: 80% EI, 20% UCB
    # Note: UCB term can be negative if mu is very good and sigma is small.
    # EI is always non-negative.
    # We want to maximize the score.
    
    # Let's try:
    # score = (0.8 * ei_term + 0.2 * ucb_term) / cost
    
    # However, if ucb_term is negative, it will lower the score.
    # This is fine, as it discourages points that are both certain and not better than f_best (which EI also handles).
    
    # Let's check the deadline.
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple EI/cost if taking too long (unlikely)
        return ei / cost
    
    scores = (0.8 * ei_term + 0.2 * ucb_term) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
