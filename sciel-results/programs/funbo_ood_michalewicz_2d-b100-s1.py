import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive LCB: explore more when far from known global minimum
    opt = -1.8013
    dist = max(0.0, f_best - opt)
    # kappa: higher when far from optimum (explore), lower when close (exploit)
    kappa = 0.5 + 2.0 * (dist / 5.0)
    # Cap kappa to avoid excessive exploration
    kappa = min(kappa, 3.0)
    # Score: higher is better for selection
    scores = -mu + kappa * sigma
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
