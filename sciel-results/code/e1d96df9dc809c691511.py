import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive beta for UCB
    # Start with moderate exploration, increase as f_best improves (gets smaller? No, f_best is the value, so it gets smaller as we improve).
    # Wait. If we improve, f_best decreases.
    # If f_best decreases, we are closer to optimum. We should exploit more (lower beta).
    # My previous logic: beta = 1 + log(1 + f_best).
    # If f_best = 25 (bad), beta = 4.3 (high exploration).
    # If f_best = 0.1 (good), beta = 1.1 (low exploration).
    # This logic holds.
    
    beta = 1.0 + np.log1p(max(0.0, f_best))
    
    scores = -mu + beta * sigma
    
    # Safety: ensure finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)
        
    return scores
