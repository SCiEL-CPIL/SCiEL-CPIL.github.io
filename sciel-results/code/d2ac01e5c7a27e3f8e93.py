import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Gaussian KDE of the posterior minimum.
    
    Strategy:
    - Instead of linear UCB or standard EI, we model the distribution of the 
      objective value at each candidate point as a Gaussian N(mu, sigma^2).
    - The score is the probability density of achieving a value close to f_best 
      (or better), weighted by the inverse cost.
    - Specifically, we use the PDF of the normal distribution evaluated at f_best.
    - This is structurally different from EI (which integrates) or UCB (which is linear).
    - The PDF peaks when mu is close to f_best and sigma is moderate. 
      If sigma is too small and mu > f_best, PDF is low. 
      If sigma is too large, PDF is spread out and lower. 
      If mu < f_best, PDF at f_best is in the tail, but we want the probability of being < f_best.
      
    Actually, for minimization, we want P(f(x) < f_best) = Phi((f_best - mu)/sigma).
    However, PI is often too greedy. 
    A "Density" based score can be interpreted as: how likely is this point to be 
    the *next* best point? 
    
    Let's use the **Expected Improvement** but derived differently? 
    No, the directive asks for a structural change.
    
    Let's use **Probability of Improvement (PI)** but with a **Cost-Effective** twist.
    Standard PI = Phi((f_best - mu) / sigma).
    Cost-aware PI = PI / cost.
    
    Why is this structurally different from the current UCB?
    UCB = (f_best - (mu - kappa*sigma)) / cost = (f_best - mu + kappa*sigma) / cost.
    PI / cost = Phi((f_best - mu) / sigma) / cost.
    
    The non-linearity of Phi is the key difference. 
    UCB is linear in mu and sigma. PI is non-linear.
    This non-linearity naturally caps the exploration bonus (as sigma -> inf, Phi -> 0.5) 
    and caps the exploitation (as mu -> -inf, Phi -> 1).
    
    This avoids the "divergence" issues of UCB where large sigma can dominate 
    regardless of mu. It is a bounded, probabilistic measure.
    
    To make it more robust and potentially better than pure PI, we can use 
    **Expected Improvement (EI)** but the prompt says the current program is UCB (score 0.967) 
    and the parent was EI-like (score 1.04). 
    Wait, the current program in the prompt is UCB. 
    The parent (p26) had score 1.0467 and used Thompson Sampling/EI hybrid? 
    The text says "Initial program... EI". 
    Inspiration 1 (1.0467) used Thompson Sampling.
    
    If I use PI, it's a different family from UCB. 
    Let's try **Cost-Aware Probability of Improvement**.
    
    score = Phi((f_best - mu) / sigma) / cost
    
    To prevent premature convergence, we might want to add a small exploration term, 
    but PI already has an exploration component (high sigma increases the chance 
    of crossing f_best if mu is not too high). 
    However, if mu is very high, high sigma helps. If mu is low, low sigma is fine.
    
    One issue: If sigma is 0, Phi((f_best - mu)/0) is a step function. 
    We should handle sigma=0.
    
    Let's also consider **Cost-Effective EI** again, but the current one failed. 
    Maybe the failure was due to the specific constants. 
    But PI is a cleaner structural change.
    
    Let's implement PI with a safety check for sigma.
    """
    start_time = time.time()
    deadline = 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive for the PDF/CDF calculation
    # If sigma is very small, the CDF becomes a step function.
    # We add a small epsilon to avoid division by zero in the Z-score.
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Z-score for Probability of Improvement: P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / safe_sigma
    
    # Calculate CDF
    pi = norm.cdf(z)
    
    # Cost adjustment
    safe_cost = np.maximum(cost, 1e-6)
    
    # Score: PI per unit cost
    score = pi / safe_cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
