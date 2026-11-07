import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division issues, though noise 1e-6 should prevent 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is 0 if mu > f_best (approximately, due to numerical precision)
    
    # Calculate Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    beta = 3.0
    ucb_term = beta * sigma_safe - mu
    
    # Blend EI and UCB term
    # EI is in "improvement" units. UCB term is in "objective" units (shifted).
    # They are not directly comparable in magnitude without scaling.
    # However, we can just use UCB as the primary driver for exploration, 
    # and EI for exploitation.
    
    # A common robust strategy is to use EI if it's significant, else UCB.
    # Or simply use UCB.
    # Given the "Stagnation" warning, let's try a pure UCB-based score 
    # to ensure we explore regions where EI is 0 but uncertainty is high.
    
    # Score = UCB_term / cost
    # But wait, if mu is very bad (e.g., 0) and f_best is -3, 
    # UCB_term = 3*sigma - 0. If sigma is 0.1, term is 0.3.
    # EI would be 0.
    # If we pick based on UCB, we explore.
    
    # Let's try a hybrid: 
    # If EI > 0, it's exploitation. 
    # If EI == 0, it's exploration.
    # A smooth blend:
    # score = (ei + ucb_term) / cost ?
    # The scales might be off.
    # Let's normalize? No, we just need ranking.
    
    # Let's stick to the inspiration: "Adaptive blending".
    # Since we can't adapt beta easily, let's try a fixed blend.
    # Actually, just using UCB is a strong structural change from EI.
    
    scores = ucb_term / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
