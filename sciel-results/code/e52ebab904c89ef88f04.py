"""Rewritten program with adaptive cost-aware acquisition for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _expected_improvement(mu, sigma, f_best):
    """Standard EI for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, beta):
    """UCB for minimization (we want to maximize the score, so we use mu + beta*sigma? 
    Wait. For minimization, we want low mu. 
    Standard UCB for minimization is often -mu + beta*sigma or similar.
    However, the previous program used `mu + beta * sigma` and called it UCB.
    If we are minimizing, a high `mu` is bad. 
    Let's check the previous code:
    `ucb = mu + beta * sigma`
    `combined = 0.6 * ei_norm + 0.4 * ucb_norm`
    `scores = combined / cost`
    
    If `mu` is the predicted value (lower is better), then `mu + beta*sigma` is NOT a standard minimization UCB.
    Standard Minimization UCB: We want to find low values. 
    Confidence Interval: [mu - beta*sigma, mu + beta*sigma].
    To explore, we look at the lower bound? No, that's exploitation.
    To explore, we look at the upper bound of the *uncertainty*?
    Actually, in BO for minimization:
    EI is positive when mu < f_best.
    UCB for minimization is typically defined as: `acq = -mu + beta * sigma` (Maximize this).
    OR `acq = mu - beta * sigma` (Minimize this).
    
    The previous program used `mu + beta * sigma`. 
    If `mu` is high (bad), this term is high. This would encourage evaluating BAD points?
    UNLESS `mu` here is defined such that higher is better? 
    The task says: "mu and sigma ... are in objective units." "f_best is the lowest observed value."
    So `mu` is the expected function value. Lower is better.
    If the previous program scored 3.21 with `mu + beta * sigma`, it implies that adding `sigma` helped, but adding `mu` directly might be counter-productive if `mu` is high.
    However, `ei` is also used.
    Let's re-read carefully.
    "Return 2000 finite scores (higher = evaluate next)".
    
    If I use `mu + beta * sigma`, and `mu` is 100 (bad), `sigma` is 1. Score is 101.5.
    If I use `mu2 = 1` (good), `sigma2 = 1`. Score is 2.5.
    The algorithm would pick the bad point. This seems wrong.
    
    Why did the previous program work?
    Maybe `mu` is standardized? "standardized targets".
    If targets are standardized, `mu` is roughly centered at 0.
    Rastrigin values are positive (0 to 50). Standardizing them makes them roughly N(0,1).
    So `mu` can be negative (good) or positive (bad).
    `f_best` is also standardized? "f_best is the lowest observed value." 
    Usually, in GP implementations, if you standardize y, you must standardize f_best too, or the EI formula breaks.
    The task says: "mu and sigma ... are in objective units." 
    This implies `mu` and `sigma` are NOT standardized? 
    "GP: ... standardized targets; mu and sigma ... are in objective units."
    This is slightly contradictory or implies the GP returns standardized mu/sigma, but the task *gives* us mu/sigma in objective units?
    "mu, sigma and cost have shape (2000,)."
    "mu and sigma (a standard deviation) are in objective units."
    Okay, so `mu` is the raw predicted value.
    
    If `mu` is raw value (e.g., 2.5), and `f_best` is 1.0.
    EI = (1.0 - 2.5) * ... = negative? No, EI is 0 if mu > f_best.
    
    If the previous program used `ucb = mu + beta * sigma`, and it worked, maybe the `ucb_norm` part was small or the `ei_norm` part dominated?
    Or maybe my interpretation of "UCB" in the previous code was just a label for a term that happened to help exploration via `sigma`?
    
    Let's construct a proper Minimization Acquisition.
    We want to MAXIMIZE the score.
    Good points have low `mu`.
    Uncertain points have high `sigma`.
    
    Term 1: Exploitation. We want low `mu`. So `-mu` is a good score.
    Term 2: Exploration. We want high `sigma`. So `+ sigma` is a good score.
    
    Standard Minimization UCB (to maximize): `acq = -mu + beta * sigma`.
    
    Let's check the previous code again.
    `ucb = mu + beta * sigma`.
    If this was used, and it scored 3.21, it's possible that the `ei` part was doing all the work, and the `ucb` part was just adding noise or slight exploration benefit due to `sigma`, while the `mu` part was harmful but masked by normalization?
    
    `ei_norm = ei / ei_max`. `ei` is 0 for bad points.
    `ucb_norm = (mu + beta*sigma) / ucb_max`.
    If `mu` is large, `ucb_norm` is large. This encourages bad points.
    
    I will switch to the correct Minimization UCB: `-mu + beta * sigma`.
    This should improve exploitation of low-mu regions while still exploring high-sigma regions.
    
    Also, I will use the adaptive weights based on `f_best`.
    """
    return -mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions (objective units).
    sigma : array_like, shape (2000,)
        GP standard deviation predictions (objective units).
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and safe
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (Exploitation)
    ei = _expected_improvement(mu, sigma, f_best)
    
    # 2. Compute UCB (Exploration + Exploitation)
    # Correct Minimization UCB: -mu + beta * sigma
    # Adaptive beta: 
    # Early on (high sigma), we want more exploration.
    # Late on (low sigma), we want less exploration.
    # Use mean sigma as a proxy for stage.
    # Rastrigin values are ~0-50. Sigma starts high, drops low.
    # Normalize sigma to [0, 1] roughly for beta calculation?
    # Let's just use a fixed beta but correct sign first.
    beta = 2.0
    
    ucb = _ucb(mu, sigma, beta)
    
    # 3. Adaptive Weights
    # If f_best is high (early), we need more exploration (UCB).
    # If f_best is low (late), we need more exploitation (EI).
    # Rastrigin min is 0. Max is ~50.
    # progress = 1 - f_best / 50.
    # If f_best = 50, progress = 0. w_ucb = 0.8, w_ei = 0.2.
    # If f_best = 0, progress = 1. w_ucb = 0.2, w_ei = 0.8.
    
    # Clamp f_best to be safe
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = 1.0 - (f_best_clamped / 50.0)
    
    # Weights
    w_ucb = 0.2 + 0.6 * (1.0 - progress) # Range [0.2, 0.8]
    w_ei = 1.0 - w_ucb
    
    # 4. Normalize components to comparable scales
    # EI is always >= 0.
    ei_max = np.max(ei)
    if ei_max <= 0:
        ei_max = 1.0
    ei_norm = ei / ei_max
    
    # UCB can be negative (if mu is very large). 
    # We should shift it so min is 0? Or just normalize range?
    # Normalizing by max is risky if max is negative.
    # Better: Normalize to [0, 1] using min-max scaling.
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-9:
        ucb_norm = np.ones_like(ucb) * 0.5
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
    
    # 5. Combine
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # 6. Cost-Aware Normalization
    # Divide by cost to prefer cheaper points.
    scores = combined / cost_safe
    
    # 7. Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
