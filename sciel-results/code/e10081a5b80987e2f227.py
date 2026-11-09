import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge case where f_best is very small (near convergence)
    if f_best <= 1e-9:
        return -mu
    
    # Ensure sigma is non-negative and safe for normalization
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Quantile normalization of sigma (rank-based)
    # This maps sigma to [0, 1] based on its relative rank among candidates.
    # This is robust to the absolute scale of sigma and ensures that
    # the exploration term is always in a consistent range [0, 1] * beta.
    n = len(sigma_safe)
    if n > 1:
        ranks = np.argsort(np.argsort(sigma_safe))
        sigma_norm = ranks / (n - 1)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Structural Change: Probability of Improvement (POI) based scoring
    # Compute the CDF of the posterior distribution at the current best value.
    # P(f < f_best) = Phi((f_best - mu) / sigma)
    # This ranks candidates by their likelihood of beating the incumbent,
    # decoupling selection pressure from the absolute scale of the objective.
    
    # To combine with exploration, we use a modified POI that also considers
    # the potential for large improvements via a dynamic exploitation/exploration balance.
    
    # Standardized improvement: z = (f_best - mu) / sigma
    # For very small sigma, z is large if mu < f_best (high POI) or small/negative if mu > f_best.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in the CDF calculation
    z = np.clip(z, -10, 10)
    
    # Compute POI: P(f < f_best) = Phi(z)
    # Using the error function: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    poi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
    
    # The POI score is bounded in [0, 1].
    # However, pure POI can be too greedy in later stages.
    # We blend POI with a rank-based exploration term to ensure diversity.
    
    # Dynamic weight for exploration based on f_best.
    # Early on (large f_best), we want more exploration.
    # Later (small f_best), we want more exploitation (higher POI weight).
    # The inspirations used beta = f_best for a linear UCB-like term.
    # Here, we scale the exploration contribution inversely to how "confident" we are.
    
    # Let's use a blend: Score = alpha * POI + (1 - alpha) * exploration_term
    # Or more simply, use POI as the primary driver but add a small exploration bonus
    # scaled by uncertainty rank to break ties and encourage exploration in high-sigma regions
    # when POI is similar.
    
    # A common heuristic is to maximize POI, but add a small term proportional to sigma
    # to avoid getting stuck in local minima where POI is high but improvement is marginal.
    
    # Let's define the score as:
    # score = poi + lambda * sigma_norm
    # where lambda is a small constant or dynamic.
    
    # Since POI is in [0,1] and sigma_norm is in [0,1], we need to balance them.
    # If f_best is large, the difference (f_best - mu) is large for most points, so POI is close to 1 for many.
    # The discrimination comes from the tail.
    
    # Alternative: Use the log-POI or a transformed version.
    # Let's stick to the direct POI plus a scaled exploration term.
    
    # The inspiration programs used: score = -mu + beta * sigma_norm
    # The current program used: score = -mu + lambda * log(1 + sigma_norm)
    
    # The directive asks for a structural change to POI.
    # Let's compute POI and use it directly, but ensure that high uncertainty regions
    # are still explored if their POI is not overwhelmingly higher.
    
    # A robust combination is:
    # score = poi * (1 + gamma * sigma_norm)
    # This boosts the POI of high-uncertainty points slightly.
    
    # Or simply:
    # score = poi + delta * sigma_norm
    
    # Let's try: score = poi + 0.1 * sigma_norm
    # The 0.1 is a small exploration bonus.
    
    # However, if f_best is very small, POI values will be very close to 0 or 1.
    # The exploration term helps navigate.
    
    # Let's use a dynamic exploration weight that decreases as f_best decreases.
    # But POI already adapts to f_best via the z-score.
    
    # Let's use a fixed small exploration weight for the rank term.
    # This ensures that even if POI is flat, we explore high sigma regions.
    
    exploration_weight = 0.1
    
    scores = poi + exploration_weight * sigma_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
