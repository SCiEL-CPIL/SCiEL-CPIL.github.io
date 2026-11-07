"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _ei_score(mu, sigma, f_best):
    """Calculate Expected Improvement scores."""
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _approx_posterior_update(mu, sigma, x_new, y_new, f_best):
    """
    Approximate the GP posterior update after evaluating x_new with value y_new.
    
    This is a heuristic approximation to avoid full matrix inversion.
    We assume the new point x_new is the point of interest and we want to estimate
    the mean and variance at other points.
    
    For a 1-step lookahead, we need to estimate:
    mu_next(x) and sigma_next(x) for all x in the candidate set.
    
    Standard GP update:
    k(x, x_new) is the covariance between candidate x and new point x_new.
    k_new = k(x_new, x_new) + noise
    
    mu_next(x) = mu(x) + k(x, x_new) * (y_new - mu(x_new)) / k_new
    sigma_next(x)^2 = sigma(x)^2 - k(x, x_new)^2 / k_new
    
    However, we don't have the kernel matrix or the specific x coordinates passed to 
    the acquisition function. We only have mu and sigma for 2000 candidates.
    
    WAIT: The problem states "No coordinates are passed." 
    This means we CANNOT compute k(x, x_new) because we don't know x.
    
    Therefore, a true GP-based lookahead is impossible without coordinates.
    
    ALTERNATIVE STRATEGY:
    Since we cannot update the GP, we must use a strategy that relies only on 
    the current mu, sigma, f_best, and cost.
    
    The current strategy is EI / cost.
    
    What other strategies exist that don't require coordinates?
    1. Probability of Improvement (PI) / cost
    2. Lower Confidence Bound (LCB) / cost
    3. UCB / cost
    4. Tuned EI (varying xi)
    5. Weighted combination of EI and PI
    
    Let's try a different design family: **Probability of Improvement (PI) with aggressive exploration**.
    PI is less sensitive to the magnitude of improvement than EI, which can help escape local 
    plateaus in the rastrigin function where EI might be small but PI is high.
    
    Or **Lower Confidence Bound (LCB)**:
    LCB = mu - beta * sigma
    We want to minimize this, so acquisition = -LCB / cost = (beta * sigma - mu) / cost.
    
    Let's try **UCB (Upper Confidence Bound)** adapted for minimization:
    We want to find low values. So we look for points with low mu and high sigma.
    Score = (mu - beta * sigma) is not right.
    For minimization, we want to minimize f(x).
    LCB(x) = mu(x) - beta * sigma(x). We want to minimize LCB.
    So acquisition = -LCB / cost = (beta * sigma - mu) / cost.
    
    Let's try a **Hybrid**: 
    The EI strategy is good. But maybe the cost normalization is suboptimal.
    The cost is 1 + mean(x_scaled). 
    Candidates near x=0 (center of domain) have lower cost.
    Candidates near corners have higher cost.
    
    Perhaps a **Tuned EI** with a small xi (optimism parameter) is better?
    Standard EI uses xi=0.
    
    Let's try **Probability of Improvement (PI)**:
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    Acquisition = PI / cost
    
    Let's try **LCB**:
    Acquisition = (beta * sigma - mu) / cost
    
    Let's implement a **Switching Strategy**:
    If sigma is very small, use EI. If sigma is large, use LCB (exploration).
    Or simply use a **Weighted Sum** of EI and PI.
    
    Actually, let's look at the "Different Strategy" requirement.
    Current: EI / cost.
    
    Let's try **Expected Feasibility** or **Simple Regret** based on quantiles?
    
    Let's try **Lower Confidence Bound (LCB)** as the primary driver.
    LCB is often more robust in early stages or with noisy data.
    
    Implementation:
    beta = 1.0 (exploration parameter)
    score = (beta * sigma - mu) / cost
    
    Let's try to combine EI and LCB?
    score = 0.5 * (EI / cost) + 0.5 * (LCB / cost)
    
    Or, let's try **Maximizing the probability of being the global minimum**?
    This is essentially PI.
    
    Let's go with **LCB (Lower Confidence Bound)** as it is a distinct acquisition function family from EI.
    
    """
    pass

def acquisition(mu, sigma, f_best, cost):
    # LCB strategy for minimization
    # We want to minimize f(x). 
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to select x with the lowest LCB.
    # So acquisition score should be high when LCB is low.
    # score = -LCB / cost = (beta * sigma - mu) / cost
    
    # Let's use a beta that balances exploration and exploitation.
    # In BO, beta is often sqrt(2 * log(n * pi^2 / (6 * delta)))
    # But here we don't have n explicitly in a way that suggests a specific schedule.
    # Let's try a fixed beta.
    
    beta = 1.5
    
    lcb = mu - beta * sigma
    # We want to minimize lcb, so we maximize -lcb
    # Normalize by cost
    score = -lcb / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
