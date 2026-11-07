"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Global Distributional Strategy:
    Instead of scoring each point independently (local view), we first identify
    the single most promising region in the search space using the GP posterior.
    We compute the Lower Confidence Bound (LCB) for minimization: 
    LCB(x) = mu(x) - kappa * sigma(x).
    The point with the lowest LCB is our "target".
    
    Then, we construct a score that encourages exploration towards this target.
    Since we don't have coordinates, we use the cost signal as a proxy for location.
    However, a more robust approach without coordinates is to use the GP's
    predictive distribution to identify high-probability regions of low function values.
    
    Let's refine the "Global Target" idea:
    1. Compute LCB = mu - kappa * sigma for all 2000 candidates.
    2. The candidate with the minimum LCB is the current "best guess" for the minimum.
    3. We want to score other candidates based on their potential to improve upon this.
    
    Actually, the standard LCB is already a valid acquisition function.
    Why did the previous attempt fail?
    The previous attempt (inspiration 2) scored 0.5031.
    The current seed scores 1.0032.
    
    Let's look at the difference.
    Seed: EI / cost.
    Current Program: -mu + 2*sigma (UCB).
    
    The instruction says: "Make a STRUCTURAL change".
    The current program uses UCB.
    The seed uses EI/cost.
    
    Let's try a different structural approach: **Expected Improvement (EI) with adaptive exploration**.
    But EI is local.
    
    What if we use **Knowledge Gradient (KG)**? Too slow for 2000 points x 30s limit?
    2000 points is small. But KG requires integration over the posterior, which is hard with a GP.
    
    Let's try **PI (Probability of Improvement)**.
    PI(x) = P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma).
    This is a pure exploitation measure.
    
    Let's try a **Hybrid of Global and Local**:
    1. Find the index `i_best` with the lowest `mu - kappa * sigma`.
    2. This point is the "anchor".
    3. Score = EI(x) + lambda * (1 - normalized_distance_to_anchor)
    But we don't have coordinates.
    
    Wait, the cost is `1 + mean(x_scaled)`.
    `x_scaled` is in [0, 1].
    `cost` is in [1, 2].
    `cost - 1` is `mean(x_scaled)`.
    This is the average of the two coordinates scaled to [0, 1].
    This gives us one piece of information about the location: the sum of the coordinates.
    It does not give the individual coordinates.
    
    However, we can use the cost to bias the search.
    If we assume the minimum is likely to be found in regions where the GP is uncertain or low,
    we can use the cost to break ties or adjust the exploration rate.
    
    Let's reconsider the **LCB** approach but with a twist.
    The LCB is `mu - kappa * sigma`.
    If we choose `kappa` dynamically based on the "age" of the search (estimated from cost?),
    we can balance exploration and exploitation.
    
    But the prompt says "No coordinates are passed".
    
    Let's look at the **Probability of Improvement (PI)** again.
    PI is often better than EI in early stages because it doesn't penalize for the magnitude of improvement, only the probability.
    
    Let's try **PI + UCB blend**.
    Score = w1 * PI + w2 * UCB
    
    Or, let's try a **Quantile-based approach**.
    Use the 10th percentile of the predicted distribution:
    q_10 = mu - 1.28 * sigma
    This is similar to LCB with kappa=1.28.
    
    Why did the current program (UCB with beta=2.0) score 1.0032?
    It's slightly better than the seed (1.0).
    The seed was EI/cost.
    
    Let's try **Pure LCB** with a smaller kappa, say 1.0 or 1.5.
    UCB with beta=2.0 is quite exploratory.
    Maybe we need more exploitation?
    
    Let's try **Expected Hypar Improvement (EHI)**? No, standard EI is fine.
    
    Let's try a **Different Representation**:
    Instead of using `mu` and `sigma` directly, let's use the **z-score** of the improvement.
    `z = (f_best - mu) / sigma`
    High `z` means high probability of improvement.
    
    Let's try scoring by **z-score** directly.
    `score = z`
    This is equivalent to PI ranking.
    
    Let's combine **PI** and **LCB**.
    `score = PI + alpha * LCB_normalized`
    
    Let's try this structural change:
    **Adaptive LCB with Cost-Aware Exploration**.
    The cost indicates how "expensive" a point is.
    If a point is cheap (low cost), we can afford to be more exploratory (higher kappa).
    If a point is expensive (high cost), we should be more exploitative (lower kappa).
    
    `kappa(x) = base_kappa * (1 + penalty * (cost(x) - min_cost) / (max_cost - min_cost))`
    Wait, higher cost should mean lower exploration?
    Or does higher cost mean we should be more careful?
    Actually, if a point is expensive, we want to be sure we pick it. So lower exploration (lower kappa).
    If a point is cheap, we can gamble more (higher kappa).
    
    So: `kappa(x) = base_kappa * (1 - weight * normalized_cost)`
    Where `normalized_cost` is in [0, 1].
    
    Let's implement this.
    1. Normalize cost to [0, 1].
    2. Compute adaptive kappa.
    3. Compute LCB = mu - kappa * sigma.
    4. Return LCB.
    
    This is a structural change: **Adaptive Exploration Rate based on Cost**.
    
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Normalize cost to [0, 1]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Adaptive Kappa
    # Base exploration rate
    base_kappa = 1.5
    # Cost weight: how much does cost affect exploration?
    # If cost is high (1.0), we want lower kappa.
    # If cost is low (0.0), we want higher kappa.
    cost_weight = 0.5
    kappa = base_kappa * (1.0 - cost_weight * cost_norm)
    
    # Ensure kappa is positive
    kappa = np.maximum(kappa, 0.1)
    
    # Compute LCB (Lower Confidence Bound for minimization)
    # We want to minimize f.
    # LCB = mu - kappa * sigma
    # Higher LCB is better (closer to 0 or negative).
    # Wait, if mu is 10 and sigma is 1, LCB = 8.
    # If mu is 1 and sigma is 1, LCB = -0.5.
    # We want to pick the one with the LOWEST potential value.
    # So we should return -LCB? Or just LCB?
    # The task says "higher = evaluate next".
    # If LCB is the lower bound, a lower LCB is better.
    # So we should return -LCB to make higher scores better.
    
    score = -(mu - kappa * sigma)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Return a simple fallback
         return -mu
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
