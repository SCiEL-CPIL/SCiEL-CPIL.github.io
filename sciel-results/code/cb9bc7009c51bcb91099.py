"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    if time.time() - _START_TIME > 320:
        # Return a neutral score if we are running out of time
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    cost = np.maximum(cost, 1e-8)
    
    # f_best is the lowest observed value
    # For minimization, we want to maximize the expected reduction in regret.
    # Regret R(x) = f(x) - f_best (if f(x) > f_best) or 0.
    # Actually, standard EI is E[max(f_best - f(x), 0)].
    # Let's stick to EI but change the cost awareness structure.
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # New Strategy: Log-Cost Penalty
    # Instead of EI / cost, we use EI - lambda * log(cost)
    # This penalizes expensive points less severely than linear division when EI is large,
    # but prevents exploring very cheap points with near-zero EI as aggressively?
    # Actually, EI/cost favors cheap points with moderate EI.
    # EI - log(cost) is different.
    
    # Let's try a different structural change: 
    # Use **Probability of Improvement (POI)** combined with **Expected Improvement** 
    # weighted by a **Cost-Efficiency** factor that is normalized by the current best cost?
    # No, we don't have current best cost passed in, only 'cost' of candidates.
    
    # Let's try **Cost-Weighted EI with Exponential Decay of Cost**
    # score = EI * exp(-alpha * (cost - 1))
    # Since cost is in [1, 2] (1 + mean(x_scaled)), cost-1 is in [0,1].
    # This creates a smooth penalty.
    
    # Or simpler: **EI / (1 + cost)**
    # The current uses EI / cost. Since cost >= 1, EI/cost <= EI.
    # EI / (1+cost) penalizes more.
    
    # Let's look at the "different design family" directive.
    # Current: EI / cost
    # Alternative: **Expected Improvement per Unit Time/Cost** is standard.
    # What if we use **Knowledge Gradient (KG)** approximation?
    # KG is hard to compute analytically for GP without integration.
    
    # Let's try **Upper Confidence Bound (UCB)** with cost-awareness.
    # UCB = mu - kappa * sigma
    # For minimization, we want low mu and high uncertainty.
    # But UCB doesn't naturally incorporate f_best as well as EI does for regret minimization.
    
    # Let's try **Cost-Aware UCB**:
    # score = (f_best - mu) + kappa * sigma - lambda * cost
    # This is a linear combination.
    
    # Let's try **Log-Expected Improvement**:
    # score = log(EI + epsilon) / log(cost + 1)
    
    # Let's go with a **Hybrid EI/POI** with **Geometric Cost Penalty**.
    # POI = Phi(z)
    # EI = ...
    # score = (0.5 * EI + 0.5 * POI * max(f_best - mu, 0)) / (cost ** 0.5)
    
    # Actually, the most robust "different" approach is often **Expected Improvement** 
    # but using **Standardized EI** divided by **Standardized Cost**?
    # We don't have stats of costs.
    
    # Let's implement **EI / (1 + cost)** vs **EI / cost**.
    # The difference is small.
    
    # Let's implement **Probability of Improvement (POI) * (f_best - mu) / cost**?
    
    # Let's try **Expected Improvement** with a **Multiplicative Cost Factor** derived from the ratio of costs.
    # Since we don't have the distribution of costs, we can't normalize.
    
    # Let's try **Bayesian Optimization with Cost-Weighted Acquisition** using **Log-Transformed EI**.
    # score = log1p(ei) / cost
    
    # Let's try **UCB** which is a different family entirely.
    # UCB for minimization:
    # We want to minimize.
    # UCB_value = mu - beta * sigma
    # We want to select the point with the lowest UCB_value.
    # So acquisition score should be high for low UCB_value.
    # score = - (mu - beta * sigma) = beta * sigma - mu
    # How to incorporate cost?
    # score = (beta * sigma - mu) / cost
    
    # Let's try UCB. It explores more aggressively than EI.
    # EI is conservative. UCB is optimistic.
    # For Powell 4D, which is smooth, EI is usually good.
    # But UCB might find the global minimum faster if the landscape is multimodal?
    # Powell is unimodal (shifted). So EI should be fine.
    
    # However, the directive asks for a MATERIALLY DIFFERENT strategy.
    # Switching from EI to UCB is a different acquisition family.
    
    # Let's implement UCB with cost awareness.
    # We need to choose beta. A common choice is time-dependent, but we don't have t.
    # We can use a fixed beta. Let's try beta = 1.0 or 2.0.
    # Given sigma is in objective units, and f_best is likely small (near 0),
    # mu and sigma are small.
    
    # Let's try: score = (f_best - mu + beta * sigma) / cost
    # This is effectively EI without the non-linearity of the normal CDF/PDF,
    # but with a linear exploration term.
    # Actually, (f_best - mu) + beta * sigma is the **Optimistic Prediction**.
    
    # Let's use this:
    # score = (f_best - mu + 1.5 * sigma) / cost
    
    # Calculate
    optimistic = f_best - mu + 1.5 * sigma
    scores = optimistic / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
