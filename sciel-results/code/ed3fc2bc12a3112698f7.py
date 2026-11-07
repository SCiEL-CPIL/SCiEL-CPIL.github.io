"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time to ensure we stay within the 330s limit
_start_time = time.time()
_DEADLINE = 310.0  # Safety margin before 330s kill switch

def acquisition(mu, sigma, f_best):
    """
    Computes scores for 2000 candidate points.
    Strategy: Batch Expected Improvement (Batch EI) Approximation.
    
    Standard EI is myopic: it scores each point based on its marginal distribution.
    In high-dimensional multimodal landscapes like Hartmann 6D, this can lead to 
    "wasting" evaluations on points that are locally uncertain but globally irrelevant,
    or failing to exploit correlations between the current best and new candidates.
    
    Batch EI (or Greedy EI with correction) considers the joint distribution of the 
    current best function value (modeled by the GP posterior at the best point) and 
    the candidate point. 
    However, we don't have the coordinates of the best point, only f_best.
    We can approximate the "Batch" aspect by modeling the improvement as:
    E[max(f_best - f(x), 0)]
    But to make it structurally different and capture the "global minimum" aspect better,
    we use a **Correlated EI** heuristic.
    
    Actually, a more robust structural change for this specific constraint (no coords) is 
    to use **Probability of Improvement (PI)** blended with **Expected Improvement (EI)** 
    but weighted by a **Confidence in the Global Minimum** term.
    
    Let's look at the failure mode of the current program (tanh-UCB):
    It saturates exploration. This is good, but it might still explore "far" regions 
    where the GP is uncertain but the prior mean is high (bad).
    
    A distinct strategy is **Expected Regret Minimization** using a **Laplacian Approximation** 
    or simply a **Non-Linear EI** that accounts for the scale of the problem.
    
    Let's try **Expected Improvement with a Dynamic Shift**.
    Standard EI: (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    The problem with standard EI in BO is that it can be too greedy.
    The problem with UCB is that it can be too exploratory.
    
    New Strategy: **Two-Stage Scoring via Virtual Batch**
    We simulate the effect of having a "virtual best" that is slightly worse than f_best 
    to encourage exploration, or better to encourage exploitation? No.
    
    Let's implement **Expected Improvement with Uncertainty Penalty** (EI-UP).
    Score = EI(x) - lambda * sigma(x)^2 / (f_best - mu(x) + sigma(x))^2 ? No, too complex.
    
    Let's go with a **Tournament-Based EI** which is a known effective heuristic for 
    balancing exploration and exploitation without arbitrary beta parameters.
    However, we must return a score for EVERY point.
    
    Alternative Structural Change: **Gaussian Process UCB with a Non-Linear, Data-Driven Beta**
    The current program uses a fixed tanh scale. 
    Let's use **Beta derived from the Improvement Potential**.
    
    Actually, the most distinct and effective method for this specific setup (2000 candidates, 
    no coords, high-dim) is **Expected Hypervolume Improvement (EHVI)** approximated for 
    a single objective, which simplifies to a specific form of EI that is more robust to 
    scale changes.
    
    But let's try a simpler, more robust structural change: **Log-Expected Improvement**.
    Standard EI has a "cliff" behavior. Log-EI smooths this.
    
    Wait, the current program is already at 1.21. The inspirations suggest 
    "Tournament-Based" and "Gaussian Process UCB with Non-Linear Confidence".
    
    Let's implement **Expected Improvement with a Dynamic Floor**.
    Instead of a fixed beta, we use the **Median of the Improvement Distribution**.
    
    Let's try **Quantile-Based Acquisition**.
    Score = mu - q * sigma, where q is the 95th percentile of the standard normal?
    That's just UCB with beta=1.64.
    
    Let's try **Expected Improvement with a Correlation Proxy**.
    Since we don't have coordinates, we can't compute true correlation.
    
    Let's revert to a very robust, distinct method: **Expected Improvement with 
    Adaptive Temperature (Simulated Annealing style)**.
    
    Score = -mu + beta(t) * sigma
    But instead of tanh, use **Exp(-sigma^2 / 2 * scale)**? No.
    
    Let's use **Probability of Improvement (PI)** as the base, scaled by the 
    **Expected Improvement**.
    Score = PI * EI
    
    PI = Phi((f_best - mu)/sigma)
    EI = (f_best - mu) * PI + sigma * phi((f_best - mu)/sigma)
    
    This product emphasizes points that are both likely to be better AND have high expected gain.
    It suppresses points with high EI but low PI (very uncertain, far away) and 
    points with high PI but low EI (very certain, but only slightly better).
    
    This is a structural change from the additive UCB/tanh model to a multiplicative 
    risk-reward model.
    """
    
    # Time check: If we are running out of time, return a fast, safe fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Pure exploitation (lowest mu)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite or negative sigma
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1e-8)
    
    # Calculate z-score
    # Avoid division by zero or very small numbers
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (PI)
    # Phi(z)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # phi(z) is the PDF
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * pi + sigma * phi_z
    
    # To ensure stability and avoid numerical issues where EI is tiny but PI is high,
    # or vice versa, we compute the product.
    # However, EI can be negative if f_best is very low? No, EI is always >= 0.
    # If mu > f_best, z is negative. PI is small. EI is small.
    
    # The product Score = PI * EI
    score = pi * ei
    
    # Normalize the score to be in a comparable range?
    # The evaluator just picks the argmax. So relative scale doesn't matter, 
    # but finite values are required.
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If all scores are 0 or very small, it might indicate a numerical issue or 
    # that f_best is extremely low. In that case, fall back to -mu to ensure 
    # we keep exploiting the best known region.
    if np.max(score) < 1e-6:
        score = -mu
        score = np.where(np.isfinite(score), score, -np.inf)
        
    return score

# EVOLVE-BLOCK-END
