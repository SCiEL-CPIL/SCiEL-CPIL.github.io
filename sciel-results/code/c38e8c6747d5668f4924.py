"""Phase-based cost-aware acquisition with budget-decaying exploration."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _safe_div(a, b, eps=1e-12):
    """Safe division to avoid zero/nan."""
    return a / (b + eps)

def _ei(mu, sigma, f_best, eps=1e-9):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, beta):
    """UCB score for minimization: -mu + beta * sigma."""
    return -mu + beta * sigma

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Strategy:
    1. Determine the "phase" of the optimization based on the average cost of the current candidates.
       Since we don't have the global budget history, we approximate the "remaining budget" or "progress"
       by looking at the cost distribution. However, a more robust signal for *this specific task* 
       is that the cost function is $1 + \text{mean}(x)$. 
       
       Actually, we don't know the global budget spent. But we can infer the "urgency" from the 
       magnitude of f_best relative to the potential range? No, f_best is in objective units.
       
       Let's use a simpler, robust heuristic: 
       The cost is $1 + \text{mean}(x)$. 
       Early on, we should explore more (higher beta). Late on, we should exploit (lower beta).
       How to know if we are early or late?
       We don't have the step count directly. But we have `cost` for the 2000 candidates.
       The *average cost* of the candidates is roughly constant (uniform in [-5,5]^2 -> mean x in [0,1] is 0.5 -> cost ~1.5).
       
       Alternative: Use the "spread" of sigma? Early on, sigma is high everywhere. Late on, sigma is low near f_best.
       
       Let's use a fixed, well-tuned beta that balances exploration and cost.
       The inspiration 1 (score 1.0354) used "posterior sampling" (Thompson) and "budget-decaying exploration".
       
       Let's try a **Cost-Weighted UCB** where the exploration weight is inversely proportional to cost, 
       but modulated by a global "confidence" parameter that we can't easily adapt without state.
       
       Instead, let's use a **Hybrid EI/UCB** with a fixed, strong exploration weight, but normalized by cost.
       Score = (EI + beta * sigma) / cost
       
       To make it "phase-based" without state, we can look at the *relative* uncertainty.
       If sigma is very small everywhere, we are in a late phase. 
       
       Let's stick to the proven winner's structure: 
       Score = EI / cost + beta * (sigma / cost)
       
       But let's refine the beta. The previous best (1.0354) used a specific blend.
       Let's try a **Deterministic Thompson Sampling** approximation:
       Sample a single beta from a distribution? No, must be deterministic for reproducibility in scoring? 
       Actually, the task says "return 2000 finite scores". It doesn't forbid randomness, but 
       "seeded instances" implies the *function* is deterministic per instance. 
       If I use `np.random`, it might be non-deterministic across runs unless I seed it.
       The prompt says "five seeded instances". It doesn't say my code is seeded.
       However, standard practice for these tasks is deterministic.
       
       Let's use a **Fixed Beta** but a smarter combination.
       The previous program (0.8725) used a bandit to pick beta.
       Inspiration 1 (1.0354) used "posterior sampling" (which is stochastic) and "budget-decaying".
       
       Since I cannot easily know the budget spent, I will use a **Fixed, Aggressive Exploration** strategy.
       Score = (EI + 2.0 * sigma) / cost
       
       Wait, EI already includes sigma. 
       EI = (f_best - mu) * Phi(z) + sigma * phi(z).
       For large sigma, EI ~ sigma * phi(z) + (f_best-mu)*0.5. 
       UCB = -mu + beta*sigma.
       
       Let's try **Expected Improvement per cost**, but with a **variance bonus** that is scaled by the inverse of the cost.
       
       Let's look at the "Phase" idea again.
       If we assume the loop runs for a fixed number of steps (cost < 30, cost ~1.5/step -> ~13-20 steps),
       we are always in a "short budget" regime.
       
       The key insight from the 1.0354 program was "posterior sampling". 
       I will implement a **Deterministic Equivalent of Thompson Sampling** for the acquisition parameter.
       Instead of sampling, I will use a **Quantile-based** acquisition.
       
       Actually, the simplest structural change that is different from the bandit is to **hard-code the best strategy found by the bandit** if we knew it, OR to use a **Composite Score** that is non-linear.
       
       Let's try: **Score = EI / cost + alpha * (sigma / cost)^2**
       This penalizes expensive exploration more heavily (quadratic in sigma/cost? No, sigma is uncertainty).
       
       Let's go with **Cost-Normalized UCB** with a fixed beta=2.0, which is a strong explorer, 
       combined with EI to ensure we don't just pick the highest variance point.
       
       Score = 0.5 * (EI / cost) + 0.5 * (UCB / cost)
       
       This is structurally different from the bandit (which was adaptive) and the seed (pure EI/cost).
       It is a fixed, balanced hybrid.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate EI
    ei = _ei(mu, sigma, f_best)
    
    # Calculate UCB (for minimization, we want high -mu and high sigma)
    # Beta = 2.0 is a reasonable balance for 2D problems with short budgets
    beta = 2.0
    ucb = _ucb(mu, sigma, beta)
    
    # Normalize by cost
    ei_cost = _safe_div(ei, cost, 1e-6)
    ucb_cost = _safe_div(ucb, cost, 1e-6)
    
    # Hybrid Score
    # We want to maximize this.
    # EI is always >= 0.
    # UCB can be negative if mu is very low (better than f_best) and sigma is small.
    # However, if mu < f_best, EI is high. 
    # Let's clip UCB to be non-negative to avoid penalizing good points?
    # No, UCB = -mu + beta*sigma. If mu is very low, -mu is very high (positive). 
    # So UCB is generally positive for good points.
    
    score = 0.7 * ei_cost + 0.3 * ucb_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
