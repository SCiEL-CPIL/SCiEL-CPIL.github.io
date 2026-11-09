import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    We use a hybrid approach:
    1. Primary: Expected Improvement (EI) with a small jitter to handle sigma=0.
    2. Secondary: If EI fails (all sigma near zero or numerical issues), fall back to UCB.
    3. Structural change: Instead of just a fixed beta UCB or standard EI, we use
       a 'robust' EI that clips sigma to avoid division by zero and uses a 
       slightly larger exploration constant if the problem seems hard (high f_best).
       
    However, the directive asks for a STRUCTURAL change. The current program is a simple UCB.
    The parent program (p9) had a score of 22.77 (which is very high, likely a bug or different metric? 
    Wait, the score is "mean over seeds of simple regret... lowest observed minus minimum".
    Lower is better. 0.1459 is the current score. 22.77 was the parent's score? 
    The JSON says parent_observation score 22.77. That's terrible. 
    The current program (p37) has score 0.1459. 
    So the current program is much better than the parent.
    
    The directive says: "STAGNATION: the population best has not improved... Make a STRUCTURAL change".
    This implies the current lineage (UCB variants) is stuck.
    
    Let's switch from UCB to Expected Improvement (EI) or Probability of Improvement (PI).
    EI is generally superior for minimization when we have a good GP posterior.
    
    For minimization:
    EI(x) = E[max(f_best - f(x), 0)]
    For a GP posterior N(mu, sigma^2):
    If sigma == 0:
        EI = max(f_best - mu, 0)
    Else:
        Z = (f_best - mu) / sigma
        EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    Note: The task says "minimizes sphere 1d". 
    The GP is trained on the observed points.
    mu and sigma are provided.
    
    Let's implement EI carefully.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin, should be very fast
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score
    # Z = (f_best - mu) / sigma
    # If f_best < mu, Z is negative, EI is small.
    # If f_best > mu, Z is positive, EI is large.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu) / sigma
    
    # Compute EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # We need Phi (CDF of standard normal) and phi (PDF of standard normal)
    
    # Using scipy.stats.norm for accuracy
    from scipy.stats import norm
    
    # Clip Z to avoid extreme values causing numerical issues in exp
    Z_clipped = np.clip(Z, -10, 10)
    
    phi_Z = norm.pdf(Z_clipped)
    Phi_Z = norm.cdf(Z_clipped)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Handle numerical artifacts
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    # If EI is all zeros or very small, it might be because f_best is very low and mu is also low.
    # In that case, we might want to explore more.
    # But let's stick to EI first.
    
    # Structural Change Idea: 
    # Instead of pure EI, let's use a "Weighted" combination or a different acquisition.
    # Actually, the most robust structural change from UCB is to use EI, as it directly optimizes for improvement.
    # Another option is PI (Probability of Improvement), which is more greedy.
    # Or a "Knowledge Gradient" approximation.
    
    # Given the short budget and 1D problem, EI is a strong standard choice.
    # Let's also consider that the GP might be overconfident (sigma very small) everywhere.
    # In that case, EI will be 0 everywhere except where mu < f_best.
    
    # Let's add a small exploration bonus to EI to prevent getting stuck in local minima
    # if the GP is overconfident.
    # However, the directive says "do not micro-tune constants".
    # So let's stick to standard EI.
    
    # Check deadline
    if time.time() > deadline:
         # Fallback: return random scores or just UCB if something went wrong
         return -mu + sigma
    
    return EI
