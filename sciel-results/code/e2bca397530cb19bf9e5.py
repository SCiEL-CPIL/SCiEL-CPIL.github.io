import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses a hybrid strategy that adapts to the progress of the optimization:
    1. Early stage (f_best is high): Use UCB with a larger beta to explore broadly.
       This helps escape the initial random points and find the basin of attraction of 0.
    2. Late stage (f_best is low): Use Expected Improvement (EI) to exploit the
       promising regions near the optimum.
    
    The transition is smooth, based on the ratio of current best to the initial
    expected best (which is roughly 25/3 = 8.33 for uniform points on [-5,5]).
    
    We also handle numerical stability for sigma=0 and extreme Z values.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score for EI
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu) / sigma
    
    # Compute EI
    # Clip Z to avoid numerical issues
    Z_clipped = np.clip(Z, -10, 10)
    
    phi_Z = norm.pdf(Z_clipped)
    Phi_Z = norm.cdf(Z_clipped)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Handle numerical artifacts
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Compute UCB
    # For minimization, UCB = mu - beta * sigma
    # We choose beta based on progress
    # Initial f_best for uniform points on [-5, 5] is around 8.33 (E[x^2] = 25/3)
    # As f_best decreases, we reduce beta to shift from exploration to exploitation
    
    # Define a progress metric: 0 when f_best is high, 1 when f_best is close to 0
    # Use a sigmoid-like function for smooth transition
    # f_best starts around 8-10 and goes to 0
    # Let's use: progress = 1 / (1 + f_best)
    # When f_best = 10, progress ~ 0.09
    # When f_best = 1, progress ~ 0.5
    # When f_best = 0.1, progress ~ 0.91
    # When f_best = 0.01, progress ~ 0.99
    
    progress = 1.0 / (1.0 + f_best)
    
    # Beta: starts at 2.0 (exploration) and decreases to 0.1 (exploitation)
    # beta = 2.0 - 1.9 * progress
    beta = 2.0 - 1.9 * progress
    
    UCB = mu - beta * sigma
    
    # Blend EI and UCB based on progress
    # When progress is low (early), use more UCB
    # When progress is high (late), use more EI
    
    # We need to normalize EI and UCB to comparable scales before blending
    # Since EI is always >= 0 and UCB can be negative, we need to be careful
    
    # Alternative: Use a weighted sum where the weights sum to 1
    # weight_ei = progress
    # weight_ucb = 1 - progress
    
    # But EI and UCB have different scales. Let's normalize them.
    
    # For EI: it's already in objective units
    # For UCB: it's in objective units (lower is better)
    
    # To blend them, we can convert UCB to a "score" that is higher when better
    # UCB_score = -UCB = -mu + beta * sigma
    
    UCB_score = -mu + beta * sigma
    
    # Now both EI and UCB_score are "higher is better"
    # But they have different scales. EI is typically small when f_best is small.
    # UCB_score can be large when sigma is large.
    
    # Let's use a different approach:
    # In early stages, UCB is more important because we need to explore.
    # In late stages, EI is more important because we need to exploit.
    
    # We can use a simple weighted average, but we need to handle the scale difference.
    
    # Idea: Use the maximum of the two, weighted by progress
    # score = progress * EI + (1 - progress) * UCB_score
    
    # But this might not work well if the scales are very different.
    
    # Alternative: Use a "soft max" or "log-sum-exp" style combination
    # Or simply switch between them based on a threshold
    
    # Let's try a smooth switch:
    # If progress < 0.5, use UCB
    # If progress > 0.5, use EI
    # With a smooth transition in between
    
    # Actually, let's use a different structural change:
    # Use a "Knowledge Gradient" approximation or a "Probability of Improvement" (PI)
    
    # PI(x) = P(f(x) < f_best) = Phi(Z)
    # PI is simpler than EI and might be more robust in early stages
    
    PI = Phi_Z
    
    # Now we have three candidates: EI, UCB_score, PI
    # Let's use a hybrid that switches based on progress
    
    # Early stage: UCB
    # Middle stage: PI
    # Late stage: EI
    
    # Define thresholds
    if progress < 0.3:
        # Early: Use UCB
        return UCB_score
    elif progress < 0.7:
        # Middle: Blend UCB and PI
        # Normalize UCB_score and PI to [0, 1] range for blending
        # UCB_score can be negative, so clip to 0
        UCB_norm = np.maximum(UCB_score, 0)
        UCB_max = np.max(UCB_norm) if np.max(UCB_norm) > 0 else 1
        UCB_norm = UCB_norm / UCB_max
        
        PI_max = np.max(PI) if np.max(PI) > 0 else 1
        PI_norm = PI / PI_max
        
        # Blend with equal weight
        return 0.5 * UCB_norm + 0.5 * PI_norm
    else:
        # Late: Use EI
        return EI
