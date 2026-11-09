import numpy as np
from scipy.stats import norm
import time

# Deadline guard constants
START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

# State tracking for adaptive schedule
# Since we don't have direct access to the loop index 't', we estimate progress
# based on f_best improvement and the current cost of candidates (which implies
# we are in a certain phase of the search, though cost is per-point).
# A more robust stateless approach: Use f_best to determine "phase".
# If f_best is high, we are early. If low, we are late.
# Powell 4D on [-4,5]^4 has min 0. Initial random points usually give f ~ 10-20.
# We define a "decay" based on how good f_best is relative to a typical start.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Strategy:
    1. Compute EI for minimization: E[max(f_best - f(x), 0)]
       EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
       where Z = (f_best - mu(x)) / sigma(x)
    2. Apply an exploration/exploitation trade-off via a 'xi' parameter.
       Standard EI uses xi=0. To encourage exploration early on, we can use
       a small positive xi or adjust the effective f_best.
       However, a more standard "progressive" approach is to reduce the
       influence of uncertainty (sigma) as we get closer to the optimum.
       
    3. Cost Awareness: Divide the expected improvement by the cost of evaluation.
       Score = EI(x) / cost(x)
       
    4. Adaptive Schedule:
       We estimate the "stage" of the optimization.
       If f_best is large, we are far from the optimum -> Explore more.
       If f_best is small, we are close -> Exploit more.
       
       We can model this by scaling the sigma term or using a dynamic xi.
       Let's use a dynamic xi that starts high (exploration) and decays to 0.
       xi = max_xi * (f_best / (f_best + scale))
       But wait, EI with xi > 0 means we only count improvement if f(x) < f_best - xi.
       This actually makes it MORE exploitative (stricter threshold).
       
       Alternative: Weight the uncertainty term directly.
       Score = (f_best - mu) * Phi(Z) + sigma * phi(Z)
       This is standard EI.
       
       To add exploration:
       We can use a "Cost-Effective EI" with a tunable parameter for the
       trade-off between expected improvement and cost.
       
       Let's stick to pure EI / Cost, but adjust the "improvement threshold"
       dynamically. 
       
       Actually, a simpler and often effective structural change for 4D smooth functions
       like Powell is to use **Probability of Improvement (PI)** or **EI** 
       but normalized by cost. 
       
       Let's implement EI with a dynamic 'xi' that decays as f_best improves.
       Early on (f_best high), we want to find *any* improvement, so xi=0.
       Late on (f_best low), we want to be sure of the improvement, so xi > 0?
       No, that's counter-intuitive for finding the global min.
       
       Let's look at the "Cost-Aware" aspect.
       Cost = 1 + mean(x_scaled). Range [1, 2].
       Center of the box (x=0.5 scaled) has cost 1.5.
       Corners have cost 1.0 or 2.0.
       The minimum of Powell is at a specific location. 
       
       Structural Change: **Bayesian Optimization with Cost-Effective EI and Adaptive Sigma Scaling.**
       
       We will compute:
       Z = (f_best - mu) / sigma
       EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
       
       Then Score = EI / cost
       
       To handle the "exploration" aspect structurally without arbitrary kappa:
       We can scale sigma by a factor that depends on the "confidence" in the GP.
       However, without 't', we use f_best.
       
       Let's try a different structural family: **Thompson Sampling-like score**
       Score = -mu + beta * sigma
       But normalized by cost.
       
       The prompt asks for a DIFFERENT strategy. The current is LCB/UCB.
       EI is a different family (probabilistic expectation vs confidence bound).
       
       Implementation of EI:
       1. Handle sigma=0 cases.
       2. Compute Z.
       3. Compute EI.
       4. Divide by cost.
       
       To improve on the baseline 1.0 (which is likely a simple EI or UCB),
       we should ensure numerical stability and proper handling of the cost term.
       
       One key insight: In 4D, the GP might have high uncertainty in unexplored regions.
       EI naturally balances this.
       
       Let's add a "progressive" element:
       As f_best decreases, the absolute improvement potential decreases.
       Dividing by cost remains constant relative to the space.
       
       We will use a slightly modified EI that includes a small exploration bonus
       proportional to sigma, but weighted by a decay factor based on f_best.
       
       Actually, standard EI is often sufficient. The current program's LCB might be
       too aggressive or too conservative depending on kappa.
       
       Let's implement standard Cost-Effective EI.
    """
    # Check time budget
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: Return inverse cost (prefer cheap points)
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive and finite
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    
    # Ensure cost is positive and finite
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best) # If mu is inf, assume it's as bad as current best
    
    # Compute Z for Expected Improvement
    # Z = (f_best - mu) / sigma
    # If sigma is very small, Z becomes large.
    # We clip Z to avoid overflow in norm.pdf/cdf, though cdf is stable.
    # norm.pdf can underflow, but it's fine.
    
    z = (f_best - mu) / safe_sigma
    
    # Clip z to a reasonable range to prevent numerical issues in pdf
    # norm.pdf(10) is ~ 0, norm.pdf(-10) is ~ 0.
    # norm.cdf is stable.
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If mu > f_best, Z is negative. Phi(Z) is small. phi(Z) is positive.
    # This correctly handles the case where the point is likely worse than best.
    
    phi_z = norm.pdf(z_clipped)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z_clipped)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    # Calculate EI
    ei = (f_best - mu) * cdf_z + safe_sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Aware Scoring
    # We want to maximize the expected improvement per unit cost.
    scores = ei / safe_cost
    
    # Structural Enhancement: Adaptive Exploration via Sigma Scaling
    # In the early stages (high f_best), the GP is less certain.
    # In the late stages (low f_best), we want to focus on the best regions.
    # Standard EI does this automatically because (f_best - mu) becomes smaller
    # as we get closer, but the sigma term remains.
    
    # However, if f_best is very close to 0, the absolute EI values become very small.
    # This might lead to numerical precision issues or flat landscapes in the score.
    # To mitigate this, we can normalize the scores by the current f_best scale?
    # No, the cost is absolute.
    
    # Let's add a small "exploration boost" that decays as f_best improves.
    # This is a structural change from pure EI to "EI + Exploration Bonus".
    # Bonus = beta * sigma / cost
    # beta starts high and decays.
    
    # Estimate progress: 
    # f_best starts around ~10-20 for Powell 4D.
    # Let's define a decay based on f_best.
    # If f_best > 5, we are early.
    # If f_best < 1, we are late.
    
    # Use a sigmoid-like decay for the exploration weight
    # weight = 1 / (1 + exp((f_best - threshold) * steepness))
    # We want weight HIGH when f_best is HIGH (early).
    # So we want weight to DECREASE as f_best DECREASES.
    # weight = 1 / (1 + exp((f_best - 5) * 0.5))
    # If f_best = 10: exp(2.5) ~ 12, weight ~ 1/13 ~ 0.07 (Low? No, we want high exploration early)
    
    # Let's reverse:
    # We want to ADD exploration when f_best is BAD (high).
    # So Exploration Weight should be HIGH when f_best is HIGH.
    # w = 1 / (1 + exp((5 - f_best) * 0.5))
    # If f_best = 10: exp(-2.5) ~ 0.08, w ~ 1/1.08 ~ 0.92 (High)
    # If f_best = 0: exp(2.5) ~ 12, w ~ 1/13 ~ 0.07 (Low)
    
    # This creates a schedule: High exploration early, low exploration late.
    
    # What is the bonus?
    # We add a term proportional to uncertainty (sigma).
    # bonus = w * gamma * sigma
    # gamma scales the bonus to be comparable to EI.
    # Typical EI magnitude: If f_best=10, sigma=1, EI ~ 1-2.
    # So gamma ~ 0.5 to 1.0.
    
    # Let's set gamma = 0.5 * f_best (to scale with the problem magnitude)
    # But f_best changes.
    # Let's use a fixed gamma relative to the initial scale.
    # Or simply: bonus = w * sigma
    
    # Let's try: Score = (EI + w * sigma) / cost
    # This effectively increases the "kappa" in an LCB-like manner, but modulated by EI.
    
    # Calculate weight
    # Threshold around 5.0 (mid-point of typical f_best range)
    # Steepness 0.5
    try:
        weight = 1.0 / (1.0 + np.exp((5.0 - f_best) * 0.5))
    except OverflowError:
        weight = 1.0 if f_best > 5.0 else 0.0
        
    # Ensure weight is finite
    weight = np.clip(weight, 0.0, 1.0)
    
    # Exploration bonus
    # We scale the bonus by a factor to ensure it doesn't dominate EI completely.
    # Let's use 0.5 * sigma as the bonus magnitude when weight is 1.
    bonus = weight * 0.5 * safe_sigma
    
    scores = (ei + bonus) / safe_cost
    
    # Final safety checks
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Ensure scores are positive? No, EI can be 0. 
    # But we want "higher = evaluate next".
    # If all EI are 0 and bonus is 0, scores are 0.
    # This is fine.
    
    return scores
