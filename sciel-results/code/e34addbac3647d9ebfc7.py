import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    # 1. Compute base Expected Improvement (EI) for all candidates
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use a small epsilon to avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Batched Greedy Selection with Diversity
    # We want to select a small batch (e.g., 5-10) of diverse points to "simulate" 
    # the next few evaluations or simply to create a diverse set of high-value candidates.
    # However, the function signature only returns scores for the 2000 candidates.
    # The evaluator picks the argmax of these scores.
    # So, the "batch" logic here is used to *score* the candidates:
    # A candidate gets a high score if it has high EI AND is far from other high-EI candidates.
    
    # Let's select a reference set of "prototypes" from the top EI candidates.
    # Sort candidates by EI descending
    sorted_indices = np.argsort(ei)[::-1]
    
    # Pick top K candidates as initial prototypes
    K = 10
    if len(sorted_indices) < K:
        prototypes = sorted_indices
    else:
        prototypes = sorted_indices[:K]
    
    # We need to map candidates to a spatial domain to compute diversity.
    # The problem states: "length scale 0.2 on coordinates scaled to [0, 1]"
    # and "mu, sigma ... in objective units".
    # Crucially, "No coordinates are passed."
    # This is a critical constraint. We CANNOT compute distances between candidates because we don't have their x-coordinates.
    # We only have mu, sigma, f_best, cost.
    
    # RE-EVALUATION:
    # Since we don't have coordinates, we cannot compute spatial diversity directly.
    # The "batched greedy" inspiration usually requires x-coordinates.
    # Without x, we are stuck with pointwise metrics or metrics derived from the GP statistics.
    
    # Alternative Structural Change:
    # Instead of pointwise EI/cost, use a **Pareto-efficient** or **Thompson Sampling**-like approach?
    # Or, use **Knowledge Gradient** approximation?
    # Or, exploit the **Cost** structure more deeply?
    
    # Let's look at the cost: cost = 1 + mean(x_scaled).
    # x_scaled is in [0,1]. So cost is in [1, 2].
    # The current program divides EI by cost.
    # Maybe the division by cost is suboptimal because it underweights points with slightly higher cost but much higher EI.
    
    # New Strategy: **Adaptive Thresholding on EI with Cost Penalty**
    # Or: **Maximize EI - lambda * cost** (Linear combination) instead of division.
    # Division penalizes high cost quadratically (relative to EI). Linear penalty is more standard in cost-aware BO.
    
    # Let's try: Score = EI - lambda * (cost - 1)
    # Where lambda is a tunable parameter. Since we can't tune, let's pick a reasonable lambda.
    # If cost is ~1.5, and EI is large, we want to pick it.
    # If EI is small, the cost penalty should deter evaluation.
    
    # Let's estimate the scale of EI.
    # f_best starts around 10-20 for Rosenbrock 2D uniform init.
    # EI can be up to f_best.
    # Cost is [1, 2].
    # Let's try a linear combination: EI / (1 + alpha * cost)
    # The current program is EI / cost.
    
    # What if we use **Upper Confidence Bound (UCB)** with cost?
    # UCB = mu + kappa * sigma
    # Cost-aware UCB = (mu + kappa * sigma) / cost? No, we want to minimize f, so we want low mu.
    # Acquisition for minimization:
    # EI is good.
    # PI (Probability of Improvement) is good.
    
    # Let's try a **Hybrid**:
    # Score = EI * (1 / cost^beta)
    # Current is beta=1.
    # If beta < 1, we are less sensitive to cost.
    # If beta > 1, we are more sensitive.
    
    # Structural Change: **Thompson Sampling**
    # Sample from the posterior at each candidate and pick the one with the lowest sample.
    # But we need to return 2000 scores.
    # We can approximate the "expected minimum" of a small batch? No, we return one score per candidate.
    
    # Let's go back to the "No coordinates" constraint.
    # The only structural change available without coordinates is changing the **acquisition metric** itself.
    # The current metric is CEI (Cost-Expected Improvement).
    # Another strong metric is **Cost-UCB** or **Cost-PI**.
    # Or, a **Normalized** EI that accounts for the uncertainty in f_best?
    
    # Let's try **Expected Regret** based on the distribution of the minimum?
    
    # Actually, there is a known issue with EI: it can be too exploitative early on if f_best is high.
    # **UCB** is often better for exploration.
    # Let's try **Cost-UCB**:
    # UCB(x) = -mu(x) + kappa * sigma(x)  (since we minimize, low mu is good, high sigma is good for exploration)
    # Wait, standard UCB for maximization is mu + kappa*sigma.
    # For minimization, we want to maximize the potential for improvement.
    # Lower bound of the GP: L(x) = mu(x) - kappa * sigma(x).
    # We want to find the point with the lowest possible L(x).
    # So, Acquisition = -L(x) = -mu(x) + kappa * sigma(x).
    # Then divide by cost.
    
    # Let's implement Cost-UCB with a fixed kappa.
    # Kappa should be large enough to encourage exploration.
    # For 2D Rosenbrock, kappa ~ 2.0 is a common starting point.
    
    # Implementation:
    # 1. Compute L = mu - kappa * sigma
    # 2. Score = -L / cost
    # 3. Clip to ensure finite.
    
    # This is structurally different from EI (which is non-linear in z) to UCB (linear in mu, sigma).
    
    kappa = 2.5
    
    # Lower bound of the posterior
    lower_bound = mu - kappa * sigma
    
    # We want to minimize the lower bound, so we maximize the negative lower bound
    ucb_score = -lower_bound
    
    # Cost-aware: divide by cost
    # To avoid division by zero (cost >= 1, so safe)
    final_scores = ucb_score / cost
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we have time, we could try to do a more sophisticated check, but UCB is fast.
    return final_scores
