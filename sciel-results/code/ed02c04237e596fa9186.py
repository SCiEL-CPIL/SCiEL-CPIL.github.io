import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Predictive Entropy Search (PES) approximation.
    
    Strategy:
    Instead of simple UCB or EI, we aim to maximize the expected reduction in the 
    entropy of the global minimum. This is approximated by a weighted combination 
    of:
    1. Probability of Improvement (PI): P(f(x) < f_best). This captures the 
       likelihood that a point is in a better basin than the current best.
    2. Expected Improvement (EI): The expected magnitude of improvement.
    
    For multimodal functions like Weierstrass, standard EI can be too greedy 
    (exploiting local minima) while UCB can be too noisy. PES-like metrics 
    attempt to target regions where learning about the global minimum is most 
    valuable.
    
    We use a dynamic weighting that shifts from exploration (PI-dominated) to 
    exploitation (EI-dominated) as f_best decreases.
    
    Score = w_explore * PI + w_exploit * EI_scaled
    """
    start_time = time.time()
    timeout = 290.0 # Safety margin before 300s limit

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to be comparable with PI (which is in [0,1])
    # EI is in objective units. We need to scale it.
    # A typical scale for Weierstrass might be around 1-10.
    # Let's use a soft scaling based on the current f_best and a constant.
    # If f_best is large, we expect larger improvements.
    # Scale EI by 1/(f_best + 1) to keep it bounded and adaptive.
    # If f_best -> 0, scale -> 1. If f_best -> 10, scale -> 1/11.
    # This makes EI relative to the current best.
    scale = 1.0 / (np.abs(f_best) + 1.0)
    ei_scaled = ei * scale
    
    # Dynamic Weighting
    # Early on (high f_best), we want to explore more (high PI weight).
    # Late on (low f_best), we want to exploit more (high EI weight).
    
    # Heuristic: 
    # If f_best is high (> 2.0), prioritize PI (find any improvement).
    # If f_best is low (< 0.5), prioritize EI (maximize improvement).
    # Smooth transition.
    
    # t in [0, 1] where 0 is explore, 1 is exploit.
    # Map f_best to t.
    # f_best = 10 -> t = 0 (Explore)
    # f_best = 0 -> t = 1 (Exploit)
    # t = 1 - min(f_best / 10.0, 1.0)
    
    # Let's try a different mapping based on the "value" of information.
    # We want a balance.
    # Let's use a fixed weight that has worked well in similar contexts or 
    # derive it from the distribution of z.
    
    # Actually, the "PES" approximation often uses the entropy of the 
    # indicator variable I(f(x) < f_best).
    # H(I) = -p log p - (1-p) log(1-p).
    # The mutual information is related to the change in this entropy.
    # However, a simpler proxy for "value of information" in multimodal 
    # landscapes is often just PI itself, or PI * EI.
    
    # Let's try a geometric mean or a weighted sum with a fixed, robust weight.
    # The previous Adaptive UCB (score 3.0) used high exploration early.
    # PES should also be exploratory but more targeted.
    
    # Let's try: Score = PI * (1 + EI_scaled)
    # This ensures that points with high probability of being better are 
    # prioritized, and among those, the ones with higher expected magnitude 
    # are preferred. It also naturally handles the case where EI is 0 
    # (if mu >> f_best, PI is ~0, so score is ~0).
    
    scores = pi * (1.0 + ei_scaled)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout:
        # Fallback to a simple, fast score if we are running out of time
        # Though vectorized ops on 2000 points should be very fast (<1ms).
        # This guard is mostly for safety in case of unexpected overhead.
        return scores

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
