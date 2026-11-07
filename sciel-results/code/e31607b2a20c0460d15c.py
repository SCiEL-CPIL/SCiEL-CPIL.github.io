import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 285:  # Safety margin
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Normalize cost for scaling
    # cost is typically in [1, 2] for this task (1 + mean(x))
    # We want to weight exploration vs exploitation based on how much cost we've "spent"
    # However, the function doesn't receive total cost, only the cost of the current candidate.
    # The "cost" array passed in is the cost of evaluating each of the 2000 candidates.
    
    # Strategy: Multi-Resolution EI
    # Term 1: Pure EI (focuses on improving the best value found, ignores cost of candidate)
    # Term 2: CEI (Expected Improvement per unit cost, focuses on efficiency)
    
    z = (f_best - mu) / (sigma + 1e-9)
    z = np.clip(z, -30, 30)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # CEI: EI divided by cost
    cei = ei / cost
    
    # To create a "Multi-Resolution" effect, we combine these.
    # Early in the run, we might want to be more exploratory (CEI helps find cheap good points).
    # Later, we want to exploit (EI finds the best improvement regardless of cost).
    # Since we don't have a global step counter, we can use the current f_best as a proxy.
    # If f_best is very high (far from optimum -3.86), we are early.
    # If f_best is close to -3.86, we are late.
    
    # Hartmann 3D minimum is approx -3.86278
    # Start f_best is likely around 0 or positive (uniform random in [0,1]^3 for Hartmann is often positive or small negative)
    # Let's define a "progress" metric.
    # A simple heuristic: If f_best is still relatively high, weight CEI more.
    # If f_best is low, weight EI more.
    
    # Let's use a sigmoid-like weight based on f_best.
    # Range of Hartmann 3D is roughly [-3.86, 0.4] or so.
    # Let's assume f_best starts around 0.5 and goes to -3.86.
    
    # We can also use the mean cost of the candidates as a proxy for "how much we are exploring expensive regions"
    # But simpler: Just blend EI and CEI.
    
    # A robust blend: 
    # score = alpha * EI + (1-alpha) * CEI
    # How to choose alpha?
    # If we use a fixed alpha, it's just a linear combination.
    # Let's try to make it adaptive to the current state of the GP.
    
    # Alternative Strategy: Knowledge Gradient (KG) approximation
    # KG is often superior to EI but computationally expensive.
    # With 2000 points, exact KG is too slow.
    # Approximate KG: EI is a good proxy.
    
    # Let's try a different angle: 
    # The current program scores 0.0488, which is very low. The seed (1.0) was just EI/cost.
    # The current program's portfolio approach failed. Why?
    # Maybe the max of [CEI, CEI_variant, UCB, PI] is noisy or picks bad points.
    # The seed program (EI/cost) scored 1.0. 
    # This suggests that simple CEI is a strong baseline.
    # The deviation from CEI in the current program (adding UCB, PI, etc.) hurt performance.
    
    # Therefore, the safest "different" strategy that might beat 0.0488 (and potentially 1.0) is to 
    # refine the CEI approach with a noise-adaptive scaling or a slight exploration boost 
    # that doesn't deviate as wildly as UCB/PI.
    
    # Let's try: CEI with a small UCB component, but weighted heavily towards CEI.
    # Or, simply return CEI but with a numerical stability fix and a slight bias towards 
    # points with lower sigma (more certain improvements) vs higher sigma (exploration).
    
    # Actually, let's look at the "Multi-Resolution" idea again.
    # If we just return CEI, we replicate the seed.
    # We need to do *something* different.
    
    # What if we use **Probability of Improvement (PI)** normalized by cost?
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # PI is more robust to the magnitude of improvement than EI.
    # In early stages, EI can be dominated by large sigma (exploration).
    # PI focuses on the likelihood of improvement.
    
    # Let's try a blend of CEI and CPI (Cost-aware Probability of Improvement).
    # This is different from the current program's max-based portfolio.
    # A convex combination is smoother and less prone to picking outliers.
    
    pi = norm.cdf(z)
    cpi = pi / cost
    
    # Blend: 0.5 * CEI + 0.5 * CPI
    # Or maybe 0.7 * CEI + 0.3 * CPI to stay close to the known good baseline.
    
    score = 0.7 * cei + 0.3 * cpi
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
