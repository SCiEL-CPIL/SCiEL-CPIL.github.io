"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) with dynamic exploration parameter xi.
    
    Strategy:
    1. Compute the standard EI score: EI = (f_best - mu) * Phi(Z) + sigma * phi(Z), where Z = (f_best - mu) / sigma.
    2. Introduce a dynamic exploration parameter xi that adapts based on the current uncertainty landscape.
       - In high-uncertainty regions (large sigma), we increase xi to encourage exploration (making the score more sensitive to potential improvements over f_best - xi).
       - In low-uncertainty regions, we reduce xi to focus on exploitation.
    3. To make this "adaptive" without access to global history, we use the distribution of sigma across the 2000 candidates.
       Specifically, we compute a robust measure of the current exploration potential, e.g., the median sigma or a percentile.
       We then scale xi relative to the gap between f_best and the minimum predicted mean (mu_min).
       
    Actually, a simpler and more robust structural change from the current Thompson Sampling (which is stochastic) is to use a deterministic
    acquisition that balances exploration and exploitation more explicitly than a single sample.
    
    Let's use a variant of EI that is more aggressive in exploration when sigma is high.
    Standard EI with xi=0 is:
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * pdf((f_best - mu)/sigma)
    
    We can modify this by introducing a term that rewards high sigma more strongly when f_best is still far from the known minimum (0).
    However, f_best is the lowest observed value. The true minimum is 0.
    
    Let's try a different approach: **Probability of Improvement (POI) + Exploration Bonus**.
    Score = POI + beta * sigma
    POI = Phi((f_best - mu) / sigma)
    
    This is similar to LCB but POI is bounded [0,1].
    
    Alternatively, let's look at the "Adaptive EI with negative IQR-based xi" mentioned in the knowledge.
    The idea is to set xi dynamically.
    
    Let's implement a hybrid:
    1. Calculate Z = (f_best - mu) / sigma.
    2. Calculate standard EI.
    3. Add a term proportional to sigma, but weighted by how "promising" the region is.
    
    Actually, the most distinct structural change from Thompson Sampling (stochastic) is to use a **deterministic** acquisition function that explicitly models the trade-off.
    
    Let's use **Expected Hypervolume Improvement (EHVI)** approximation or simply **Adaptive LCB**.
    
    Let's go with **Adaptive LCB**:
    Score = mu - kappa * sigma
    
    Where kappa is adaptive.
    If we use a fixed kappa, it's standard LCB.
    To make it adaptive, we can scale kappa based on the ratio of the current best improvement to the uncertainty.
    
    Let's try **Upper Confidence Bound (UCB) for minimization**:
    Score = mu - beta * sigma
    
    We want to maximize this score.
    
    To make it structurally different and potentially better, let's use **Entropy-based acquisition** or **Information Gain**.
    However, computing mutual information is expensive.
    
    Let's stick to a robust, fast, deterministic method: **Adaptive Expected Improvement**.
    
    We will compute EI with a dynamic xi.
    xi is chosen such that it encourages exploration in the early stages (when f_best is large) and exploitation later.
    Since we don't know the step count, we use f_best as a proxy.
    If f_best is close to 0, we exploit. If f_best is large, we explore.
    
    xi = max(0, f_best - mu_min) * alpha? No.
    
    Let's use a simpler heuristic:
    Compute the median sigma.
    If the median sigma is high, we are in an exploratory phase.
    
    Actually, the simplest and most effective deterministic acquisition is often just **EI** with a small positive xi to avoid getting stuck in local minima, or **LCB** with a tuned beta.
    
    The current program is Thompson Sampling (stochastic).
    The inspiration 1 (score 3.46) used EI.
    The inspiration 2 (score 6.79) is unknown but high.
    
    Let's implement **Adaptive EI**:
    1. Compute Z = (f_best - mu) / sigma.
    2. Compute EI.
    3. To encourage exploration, we add a term: exploration_bonus = sigma * (1 / (1 + (f_best / 10)**2)).
       This bonus is large when f_best is large (early optimization) and small when f_best is small (late optimization).
    
    This is a structural change: it's a deterministic, adaptive combination of EI and an uncertainty bonus.
    
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12) # Avoid division by zero
    
    # Compute Z-score
    Z = (f_best - mu) / sigma
    
    # Compute Phi(Z) and phi(Z)
    # Using scipy.stats.norm is convenient, but we can use numpy for speed if needed.
    # scipy is available.
    from scipy.stats import norm
    
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Standard EI
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Adaptive Exploration Bonus
    # We want to explore more when f_best is high (far from optimum 0)
    # Scale factor: decays as f_best approaches 0
    # Let's use a soft decay: weight = 1.0 / (1.0 + (f_best / 1.0)**2)
    # This is ~1 when f_best < 1, and ~0 when f_best > 10.
    # The Beale function starts around ~15-20 for random points in [-4.5, 4.5]^2.
    # So initially f_best will be large.
    
    weight = 1.0 / (1.0 + (f_best / 1.0)**2)
    
    # The bonus is proportional to sigma
    # We need to tune the coefficient. Let's call it beta_explore.
    # If beta_explore is too high, we just pick the point with highest sigma.
    # If too low, it's just EI.
    # Let's set beta_explore to a moderate value, e.g., 0.5 * weight.
    # Actually, let's make the bonus relative to the scale of EI.
    
    # Alternative: Just use LCB with adaptive beta.
    # Score = mu - beta * sigma
    # beta = base_beta * weight
    
    # Let's combine EI and LCB.
    # Score = EI + weight * sigma
    
    # This ensures that even if EI is 0 (mu > f_best), we still have a positive score if sigma is high and weight is high.
    # This encourages exploring high-uncertainty regions early on.
    
    scores = EI + weight * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    if time.time() - start_time > 290.0:
        return -mu
        
    return scores
# EVOLVE-BLOCK-END
