# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using soft-thresholded improvement probability.
    
    Strategy:
    1. Compute a bounded "probability of improvement" signal using:
       p_imp = 1 - exp(-lambda * max(0, f_best - mu))
       This is less sensitive to outlier GP predictions than raw EI.
    2. Multiply by a confidence term that increases with uncertainty:
       conf = 1 - exp(-gamma * sigma)
       This encourages exploration in high-uncertainty regions.
    3. Combine into a score: score = p_imp * conf / cost
       This creates a cost-efficiency score that balances exploitation (low mu)
       and exploration (high sigma) while penalizing expensive evaluations.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    scores: (2000,) array of finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Soft-thresholded improvement probability
    # lambda controls how quickly the probability saturates
    lam = 1.0
    improvement = np.maximum(0.0, f_best - mu)
    p_imp = 1.0 - np.exp(-lam * improvement)
    
    # Confidence/exploration term based on uncertainty
    # gamma controls how strongly we reward high uncertainty
    gamma = 2.0
    conf = 1.0 - np.exp(-gamma * sigma)
    
    # Combine and divide by cost
    score = (p_imp * conf) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
