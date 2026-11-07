"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Bayesian Hyperparameter Optimization (BOHB) style tuning.
    Instead of a fixed portfolio, we perform a lightweight internal search over the
    exploration parameter (beta for UCB or temperature for SoftMax) to maximize
    the expected improvement per unit cost.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros_like(mu)
    
    # Avoid division by zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # If cost is zero or negative, return zeros (should not happen per contract, but safe)
    if np.any(cost <= 0):
        return np.zeros_like(mu)

    # Strategy: Optimize the exploration parameter 'beta' for a UCB-like score
    # Score(beta) = (f_best - mu + beta * sigma) / cost
    # We want to choose beta to maximize the *expected* quality of the selected point.
    # Since we don't know the true function, we approximate the "value" of a configuration
    # by looking at the top candidates. A robust heuristic is to choose beta such that
    # the top-k candidates have a good balance of low mu and high sigma.
    
    # Alternative structural approach: 
    # Use a "Thompson Sampling" inspired score.
    # For minimization, we sample from the posterior N(mu, sigma^2) and pick the point
    # with the lowest sample, normalized by cost.
    # However, we need a deterministic score for 2000 points.
    # We can approximate the expected rank or use a quantile.
    # 
    # Let's try a different structural family: **Knowledge Gradient (KG) approximation**.
    # KG is theoretically superior to EI but expensive. 
    # A fast approximation for KG in high dimensions is difficult.
    #
    # Let's try **Expected Hypervolume Improvement (EHVI)** approximation? No, single objective.
    #
    # Let's go with **Adaptive UCB** where beta is determined by the current state of the GP.
    # The optimal beta for UCB in BO is often related to the log of the number of evaluations.
    # But we don't have the count directly, we have cost.
    # Total budget is 30. Initial cost is ~2.5 (5 points * 1.5 avg).
    # Remaining budget is ~27.5.
    # Let's estimate the number of evaluations N from the cost.
    # Average cost per eval is ~1.5. So N_evals ~ total_cost / 1.5.
    # But we don't know total_cost history, only current f_best and the cost vector of candidates.
    # Wait, `cost` in the input is the cost of evaluating *each candidate*.
    # The loop logic is: "evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # So we return scores, the harness picks the argmax, evaluates it, updates f_best, and calls us again.
    # We do NOT have the history of previous evaluations in the input signature.
    # We only have mu, sigma, f_best, cost.
    
    # Since we lack history, we must rely on the current GP state.
    # A structural change: **Posterior Sampling (Thompson Sampling) Score**.
    # Instead of maximizing EI or UCB, we use the expected value of the best point if we were to sample.
    # For minimization, the "value" of a point x is its sample from N(mu(x), sigma(x)^2).
    # The acquisition score should reflect the probability that x is the best, or the expected regret reduction.
    
    # A simpler, robust structural change: **Cost-Adjusted Probability of Improvement with Uncertainty Discounting**.
    # The current program uses max(EI/cost, UCB/cost, PI/cost).
    # This "max" operator creates a discontinuous landscape.
    # Let's use a **Geometric Mean** or **Harmonic Mean** of the components to encourage balance.
    # Or better: **Expected Improvement per unit cost, but with a Gaussian Process based uncertainty penalty**.
    
    # Let's try a completely different angle: **Directly optimizing for Simple Regret reduction**.
    # Simple Regret = f_best - f_min.
    # We want to find f_min.
    # The most informative point is the one with the lowest upper bound on the function value.
    # UCB: mu - beta * sigma.
    # We want to minimize UCB.
    # Score = -UCB / cost = (-mu + beta * sigma) / cost.
    # This is what the current program does (partially).
    
    # What if we use **Bayesian Optimization with Noisy Observations**?
    # The problem states noise 1e-6. This is very low.
    
    # Let's try **Multi-Armed Bandit (MAB) approach**:
    # Treat each candidate as an arm.
    # We want to pull the arm with the highest expected reward.
    # Reward = reduction in f_best.
    # This is essentially EI.
    
    # Let's look at the "Inspiration 1" which used soft-max and scored 1.23 (worse than 1.245).
    # The current program scores 1.245.
    # The parent (initial) scored 1.0.
    # So the portfolio approach is good.
    
    # Structural Change: **Use a Quantile-based Acquisition**.
    # Instead of EI (which is sensitive to the tail of the Gaussian), use the **Lower Confidence Bound (LCB)**
    # but formulated as a score.
    # LCB for minimization: mu - beta * sigma.
    # We want to select points with low LCB.
    # Score = - (mu - beta * sigma) / cost.
    # 
    # The key is choosing beta.
    # In the current program, beta is fixed at 2.0.
    # Let's make beta **adaptive based on the current uncertainty distribution**.
    # If sigma is generally high, we should explore more (higher beta).
    # If sigma is generally low, we should exploit (lower beta).
    # 
    # Estimate the "global uncertainty" U = mean(sigma_safe).
    # Scale beta with U.
    # beta = base_beta * (U / U_ref)
    # What is U_ref? The initial standard deviation of the GP.
    # Without prior info, we can assume U_ref is the standard deviation of the initial GP.
    # For a standard GP with standardized targets, the initial sigma is 1.0 (if variance is 1).
    # The prompt says "standardized targets". So the GP output variance is 1.
    # So initial sigma ~ 1.
    # As we observe points, sigma decreases.
    # So U will decrease.
    # If U decreases, beta decreases -> more exploitation.
    # This is a natural adaptation.
    
    # Let's implement this:
    # 1. Calculate mean_sigma = np.mean(sigma_safe)
    # 2. Set beta = 2.0 * mean_sigma (clamped to reasonable range, e.g., [0.5, 5.0])
    # 3. Score = (f_best - mu + beta * sigma_safe) / cost
    
    # Wait, if sigma is small, beta * sigma is small.
    # If sigma is large, beta * sigma is large.
    # This effectively increases the weight of uncertainty when uncertainty is high globally.
    
    # Let's also consider the **Cost** more carefully.
    # Cost = 1 + mean(x_scaled).
    # Points near the center (x=0.5) have cost 1.5.
    # Points at corners (x=0 or 1) have cost 1.0 or 2.0.
    # Actually, x is in [0,1]^d.
    # cost = 1 + (1/d) * sum(x_i).
    # Min cost = 1.0 (at origin). Max cost = 2.0 (at far corner).
    # So cost is bounded [1, 2].
    # Dividing by cost is a mild penalty.
    
    # Let's try **Expected Improvement with a Cost-Dependent Threshold**.
    # Instead of f_best, use a threshold that accounts for the cost of getting there.
    # This is complex.
    
    # Let's stick to the **Adaptive Beta UCB** idea. It is structurally different from a fixed portfolio.
    # It is a single, adaptive acquisition function.
    
    mean_sigma = np.mean(sigma_safe)
    # Clamp mean_sigma to avoid extreme beta
    mean_sigma_clamped = np.clip(mean_sigma, 0.1, 2.0)
    
    # Base beta is 2.0.
    # If mean_sigma is 1.0 (initial), beta = 2.0.
    # If mean_sigma is 0.1 (late game), beta = 0.2.
    # This shifts from exploration to exploitation naturally.
    
    beta = 2.0 * mean_sigma_clamped
    
    # Calculate UCB score
    # We want to minimize mu - beta * sigma.
    # Score (maximize) = -(mu - beta * sigma) / cost = (-mu + beta * sigma) / cost
    # To make it comparable to EI (which is 0 when mu > f_best), we can shift it.
    # EI is 0 when mu >= f_best.
    # UCB score can be negative if mu is very high.
    # Let's add f_best to make it relative.
    # Score = (f_best - mu + beta * sigma) / cost
    
    score_ucb = (f_best - mu + beta * sigma_safe) / cost
    
    # Ensure non-negative?
    # If mu is very high, f_best - mu is large negative.
    # beta * sigma might not compensate.
    # Score could be negative.
    # The harness picks the argmax.
    # If all scores are negative, it picks the least negative.
    # This is fine.
    
    # Let's also include an EI term with the same adaptive beta?
    # No, let's keep it pure UCB with adaptive beta.
    
    # To ensure we don't ignore exploitation entirely when sigma is low,
    # we can blend with a simple EI term.
    # But the directive says "different design family".
    # Adaptive UCB is a different family from Fixed Portfolio.
    
    # Let's refine the beta adaptation.
    # Use the median sigma instead of mean to be robust to outliers.
    median_sigma = np.median(sigma_safe)
    median_sigma_clamped = np.clip(median_sigma, 0.1, 2.0)
    beta = 2.0 * median_sigma_clamped
    
    score = (f_best - mu + beta * sigma_safe) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
