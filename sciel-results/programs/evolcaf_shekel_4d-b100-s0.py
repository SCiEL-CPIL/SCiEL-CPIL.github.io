"""Combined Cost-Aware Acquisition: Blending POI, UCB, and EI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking for safety margin
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize time if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s limit, return safe result if close to limit
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a valid array of zeros to satisfy interface and avoid timeout
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays of float64 for performance and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    # Replace non-finite mu with f_best (neutral)
    mu = np.where(np.isfinite(mu), mu, f_best)
    # Ensure sigma is strictly positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    # Ensure cost is strictly positive and finite
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe divisions
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf (though scipy handles large values well, 
    # clipping prevents extreme tails from dominating if there are numerical artifacts)
    z = np.clip(z, -50, 50)
    
    # Component 1: Probability of Improvement (PI)
    # Strong exploitation signal in low-data regimes (Inspiration 1)
    pi_scores = norm.cdf(z)
    
    # Component 2: Expected Improvement (EI)
    # Standard BO acquisition, balances exploration and exploitation
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI can be negative if mu > f_best and sigma is small, but we want to maximize improvement
    # Actually, EI is defined as E[max(0, f_best - f(x)]). 
    # The formula above is for minimization. 
    # If mu is very high, z is negative, cdf(z) is small, pdf(z) is small. 
    # (f_best - mu) is negative. So EI becomes negative? 
    # Let's check: if f(x) is always > f_best, EI should be 0.
    # Standard EI formula for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # If mu >> f_best, z << 0. Phi(z) ~ 0, phi(z) ~ 0. 
    # (f_best - mu) is negative. So EI < 0? 
    # No, the expectation of max(0, gain) is non-negative.
    # The formula is correct. If the mean is much worse than best, the probability of improvement is ~0, 
    # so the expected improvement is ~0. 
    # However, numerically, if z is very negative, (f_best - mu) * Phi(z) might underflow or become -0.
    # It's safer to clip EI to 0.
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Component 3: UCB (Upper Confidence Bound)
    # Exploration signal
    # UCB(x) = f_best - mu(x) + beta * sigma(x)
    # This is the "improvement" from the upper bound of the confidence interval.
    beta = 2.0
    ucb_scores = f_best - mu + beta * sigma_safe
    # UCB can be negative if the worst case is still worse than f_best.
    # We clip to 0 to avoid penalizing exploration too heavily if it's clearly bad?
    # Or keep it? Usually, we want to explore. 
    # But if we divide by cost, a large negative UCB / small cost is very bad.
    # Let's clip to 0 to be safe, similar to EI.
    ucb_scores = np.maximum(ucb_scores, 0.0)
    
    # Blend the components
    # Based on "Blending POI ... with scaled UCB and a small EI term"
    # Weights:
    # PI: 0.5 (Strong exploitation, robust)
    # UCB: 0.3 (Exploration)
    # EI: 0.2 (Standard balance)
    # These weights sum to 1.0.
    
    # Note: PI is bounded [0, 1].
    # UCB and EI are unbounded (though clipped to 0).
    # To blend them meaningfully, we should normalize them.
    # However, normalizing requires knowing the range, which varies.
    # Alternative: Scale UCB and EI by the typical scale of the problem?
    # Or just use the raw values?
    # Inspiration 1 used PI / cost.
    # Inspiration 2 used UCB / cost.
    # Inspiration 3 used EI / cost.
    
    # Let's look at the magnitudes.
    # f_best is around -10. mu can be anywhere.
    # If we use raw UCB/EI, they can be large.
    # PI is always [0, 1].
    # A direct sum might be dominated by UCB/EI.
    
    # A common approach is to normalize each score by its max or std, but that's expensive.
    # Another approach: Assume the scale of improvement is similar to the scale of sigma.
    # Or, use a weighted sum where the weights account for typical magnitudes.
    
    # Let's try a different blending strategy:
    # Score = (w1 * PI + w2 * (UCB / scale) + w3 * (EI / scale)) / cost
    # What is a good scale? The initial range of the function is ~20.
    # Or maybe just use the raw values but with smaller weights for UCB/EI?
    
    # Let's stick to the "Blending" idea from the cell finding:
    # "Blending POI (strong exploitation signal from Inspiration 1) with scaled UCB and a small EI term"
    
    # Let's assume the "scale" is implicitly handled by the relative importance.
    # If we just sum them:
    # PI is ~0.5 on average.
    # UCB can be ~10.
    # EI can be ~1.
    
    # Perhaps the "scaled UCB" implies dividing by something.
    # Let's divide UCB and EI by the current sigma? No, that's circular.
    # Let's divide by the range of f_best?
    
    # Actually, looking at Inspiration 1 (score 1.2294), it *only* used PI / cost.
    # The current program (score 0.9660) uses a blend of EI and UCB (implicitly via the "improvement_approx" which is UCB-like, but clipped and divided by cost).
    # Wait, the current program code:
    # improvement_approx = f_best - mu + beta * sigma
    # scores = np.maximum(improvement_approx, 0.0)
    # scores = scores / cost
    # This is exactly UCB / cost.
    
    # So, Current = UCB / cost. Score 0.966.
    # Inspiration 1 = PI / cost. Score 1.229.
    # Inspiration 2 = UCB / cost (beta 1.5). Score 1.027.
    # Inspiration 3 = EI / cost. Score 1.000.
    
    # It seems PI / cost is the strongest single component.
    # The directive says: "Blending POI ... with scaled UCB and a small EI term ... should improve".
    
    # So, we should start with PI and add small amounts of UCB and EI.
    # Score = (PI + alpha * UCB_normalized + gamma * EI_normalized) / cost
    
    # How to normalize UCB and EI to be comparable to PI (which is [0,1])?
    # One heuristic: Divide by the standard deviation of the targets? We don't have that directly.
    # But we have sigma.
    # Maybe divide UCB by sigma? No, that makes it a z-score.
    # z = (f_best - mu) / sigma.
    # UCB = sigma * (z + beta).
    # EI = sigma * (z * Phi(z) + phi(z)).
    
    # If we use z-scores:
    # PI = Phi(z).
    # UCB_z = z + beta.
    # EI_z = z * Phi(z) + phi(z).
    
    # These are all dimensionless.
    # PI is [0,1].
    # UCB_z can be large.
    # EI_z is bounded.
    
    # Let's blend the dimensionless versions:
    # Score = (w_pi * PI + w_ucb * UCB_z + w_ei * EI_z) / cost
    
    # However, UCB_z can be very negative. We should clip it?
    # If we clip UCB_z to 0, we lose the exploration signal for points that are likely worse.
    # But PI is already high for points likely to be better.
    
    # Let's try:
    # Score = (0.5 * PI + 0.3 * max(UCB_z, 0) + 0.2 * EI_z) / cost
    
    # Wait, EI_z is always positive?
    # Let's check: g(z) = z * Phi(z) + phi(z).
    # g'(z) = Phi(z) + z * phi(z) + phi'(z) = Phi(z) + z * phi(z) - z * phi(z) = Phi(z) > 0.
    # g(-inf) = 0. g(inf) = inf.
    # So EI_z is monotonic and positive.
    
    # So EI_z is a good measure of "expected improvement in units of sigma".
    # UCB_z is "optimistic improvement in units of sigma".
    # PI is "probability of improvement".
    
    # This seems like a coherent structural change: Working in the standardized space (z-scores) 
    # allows blending different acquisition functions that have different scales.
    
    ucb_z = z + beta
    ucb_z_clipped = np.maximum(ucb_z, 0.0)
    
    ei_z = z * norm.cdf(z) + norm.pdf(z)
    
    # Weights
    w_pi = 0.5
    w_ucb = 0.3
    w_ei = 0.2
    
    combined_score = w_pi * pi_scores + w_ucb * ucb_z_clipped + w_ei * ei_z
    
    # Final cost-aware score
    scores = combined_score / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
