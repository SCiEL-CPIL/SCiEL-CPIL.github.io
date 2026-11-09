import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 20s safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a Thompson Sampling-inspired approach 
    with Laplace correction for the posterior mean.
    
    Instead of maximizing Expected Improvement (EI), we simulate the "expected 
    next value" by adjusting the GP mean with a term that encourages exploration 
    proportional to uncertainty, similar to how Thompson Sampling samples from 
    the posterior. However, since we need deterministic scores for the top candidate, 
    we use a variant of the "Probability of Improvement" or a modified EI that 
    behaves more like a randomized search in expectation.
    
    A more direct structural change: Use **UCB (Upper Confidence Bound)** with 
    an adaptive beta that increases over time, or **Thompson Sampling** approximation.
    
    Since we cannot maintain state easily across calls without a global object 
    (and the prompt implies a stateless function signature), we implement a 
    **stateless Thompson Sampling approximation** by using a specific 
    transformation of the GP parameters.
    
    Actually, the most robust structural change that is stateless and often 
    outperforms EI in practice for multi-modal functions is **UCB (Upper 
    Confidence Bound)** with a carefully tuned exploration parameter, or 
    **Expected Improvement with a "noise" term** that simulates the variance 
    of the sampling process.
    
    Let's use a **Hybrid UCB + EI** approach or simply **UCB**.
    UCB: a(x) = mu(x) + beta * sigma(x)
    For minimization, we want to find low values. So we minimize the 
    "Upper Confidence Bound" of the *negative* objective? No, UCB for 
    minimization is: a(x) = mu(x) - beta * sigma(x).
    Wait, if we want to minimize f(x), and we have a GP with mean mu and 
    std sigma, the "worst-case" value (high value) is mu + beta*sigma. 
    The "best-case" value (low value) is mu - beta*sigma.
    To find the minimum, we should pick the point with the lowest 
    "Lower Confidence Bound" (LCB).
    
    LCB(x) = mu(x) - beta * sigma(x)
    
    The current program uses EI. EI is generally good, but LCB can be 
    better for exploration in early stages or for functions with multiple 
    peaks. The "stagnation" signal suggests the current EI might be 
    converging to a local minimum or a specific point repeatedly.
    
    Let's try **Thompson Sampling** via a deterministic proxy: 
    The expected value of the minimum of N samples from N(mu, sigma^2) 
    is not trivial, but for N=1, it's just mu.
    
    Let's switch to **UCB/LCB** with a dynamic beta. Since we don't have 
    access to the iteration count t, we can estimate the "stage" from 
    f_best and the distribution of mu/sigma? No, that's risky.
    
    Alternative Structural Change: **Knowledge Gradient (KG)** approximation.
    KG is theoretically superior to EI but computationally expensive. 
    However, there are fast approximations.
    
    Let's stick to a simpler but structurally different family: **UCB (Lower 
    Confidence Bound for minimization)**.
    
    a(x) = mu(x) - beta * sigma(x)
    
    What beta? A common choice is beta = 2 * sqrt(2 * log(N)) where N is 
    the number of points. Here N=2000.
    log(2000) ~ 7.6. sqrt(15.2) ~ 3.9. beta ~ 7.8.
    
    Let's try a fixed beta around 2.0 to 4.0, or use a standard deviation 
    based scaling.
    
    Another option: **Probability of Improvement (PI)** with a small xi.
    PI(x) = P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    
    EI is a smooth version of PI. If EI is stagnating, maybe the 
    "greediness" is the issue. UCB is more exploratory.
    
    Let's implement **LCB (Lower Confidence Bound)** for minimization.
    
    Parameters:
    mu : mean
    sigma : std
    f_best : current best
    
    Score: lower is better for the objective, but the function must return 
    scores where **higher** means evaluate next.
    So if we use LCB = mu - beta * sigma, we want to MAXIMIZE the negative 
    of the objective? No.
    We want to find the x that minimizes f(x).
    If we use LCB as a proxy for the "potential minimum", we want to pick 
    the x with the **lowest** LCB value.
    But the acquisition function must return scores where **higher** is 
    better.
    So we should return: -LCB(x) = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x).
    
    Let's verify:
    If mu is low (good) and sigma is high (uncertain), -mu is high, beta*sigma 
    is high. Score is high. Good.
    If mu is low and sigma is low, -mu is high, beta*sigma is low. Score is 
    moderately high.
    If mu is high (bad) and sigma is high, -mu is low (negative), beta*sigma 
    is high. Score might be low or high depending on magnitudes.
    
    This is a standard UCB/LCB approach.
    
    To make it structurally different and potentially more robust, we can 
    combine it with an EI term, or just use UCB. Given the stagnation, 
    UCB's explicit exploration term (beta * sigma) might help escape local 
    minima more effectively than EI's implicit exploration.
    
    Let's use a beta that is not too large to avoid pure random search, 
    but significant. beta = 2.0 is a common starting point.
    
    We will also add a time guard.
    """
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a fallback score, e.g., based on mu only (greedy)
        # Higher score for lower mu
        return -mu
    
    # Calculate LCB score
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to pick x with min LCB.
    # Score = -LCB(x) = -mu(x) + beta * sigma(x)
    
    beta = 2.0  # Exploration parameter
    
    scores = -mu + beta * sigma
    
    # Ensure finite values
    # sigma is standard deviation, should be non-negative.
    # mu should be finite.
    
    # To handle potential numerical issues where sigma is very small,
    # the term beta*sigma vanishes, and we fall back to -mu (greedy).
    
    return scores
