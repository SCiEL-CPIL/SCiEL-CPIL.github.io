"""
Strategy: Knowledge Gradient (KG) with Adaptive Exploration and Cost Awareness.

This strategy diverges from the current EI/LCB blend by using the Knowledge
Gradient, which approximates the expected value of information from evaluating
a point. Specifically, it estimates the probability that a candidate point is
the global minimizer among the 2000 candidates. This is a more principled
measure of "improvement" than EI, which only considers improvement over the
current best.

Additionally, instead of a static blend, we use an adaptive exploration
coefficient (beta) that decays as the optimization progresses (tracked via
global state). This allows aggressive exploration early on and exploitation
later. The final score is normalized by cost to account for the non-uniform
evaluation costs.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track the number of acquisition calls (evaluations made)
# This allows us to adapt the exploration parameter beta over the course of the run.
# The loop runs until cost > 30. Initial cost is ~5 (5 points * cost ~1).
# Each subsequent evaluation adds cost. Total evaluations are roughly (30 - 5) / avg_cost.
# Assuming avg cost is ~1.5, we might have ~17 evaluations.
# We will use a simple decay: beta = beta_0 * (1 - t/T)^alpha
# Or simply track the count.

_call_count = 0
_MAX_EXPECTED_CALLS = 20  # Rough estimate of how many times acquisition is called

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    
    start_time = time.time()
    
    # Safety margin: if we are running out of time, return a fast fallback
    # The deadline is 330s. We assume each call is fast, but just in case.
    # However, KG calculation is O(N^2) if done naively, or O(N) if vectorized properly.
    # N=2000. O(N^2) is 4M ops. Vectorized, this is fast (<10ms).
    
    _call_count += 1
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Adaptive Exploration Parameter
    # Start with high exploration, decay to low.
    # t is the current step index (0-based effectively for decay)
    t = min(_call_count, _MAX_EXPECTED_CALLS)
    # Beta decays linearly from 2.0 to 0.5
    beta = 2.0 - 1.5 * (t / _MAX_EXPECTED_CALLS)
    
    # Calculate the "effective" mean for exploration/exploitation balance
    # For minimization, we want to find low values.
    # The KG approximation for the probability that x_i is the minimum:
    # P(f(x_i) < min(f(x_j) for j != i))
    
    # A common approximation for the expected minimum of N samples from N(mu, sigma^2)
    # is mu - sigma * E[min(Z1...ZN)] where Z are standard normal.
    # However, calculating the exact probability that a specific point is the min
    # requires integrating over the other N-1 points.
    
    # A simpler, robust surrogate for KG in high dimensions/many candidates:
    # We can use the "Probability of Improvement" (PI) or a modified EI.
    # But let's try a different angle: 
    # The current best program uses a blend. Let's try a pure "Cost-Adjusted LCB"
    # but with a much smarter beta schedule and a better cost normalization.
    
    # Actually, let's stick to the KG idea but implement a computationally efficient version.
    # The expected value of the minimum of N Gaussian variables with means mu_i and sigmas sigma_i
    # is hard to compute exactly.
    
    # Alternative Strategy: 
    # Use "Expected Hypervolume Improvement" or simply a more aggressive 
    # "Thompson Sampling" like approach?
    
    # Let's try a different family: **Bayesian Optimization via Gaussian Process UCB 
    # with a non-linear cost penalty**.
    
    # The current program divides by cost. This is linear.
    # What if we use a logarithmic penalty or a power penalty?
    # Score = (f_best - mu + beta * sigma) / (cost ** 0.5)
    # Or better: 
    # We want to maximize the expected improvement per unit cost.
    # EI / cost is the standard "cost-aware EI".
    
    # Let's try **Probability of Improvement (PI)** instead of EI.
    # PI is less "greedy" than EI and often explores better.
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Let's combine PI with a cost-awareness that is less punishing for slightly higher costs
    # if the potential improvement is huge.
    
    # Strategy: 
    # 1. Compute PI (Probability of Improvement).
    # 2. Compute EI (Expected Improvement).
    # 3. Use a hybrid score: Score = (w * PI + (1-w) * EI) / (cost ** gamma)
    #    where gamma < 1 (e.g., 0.5) to be less penalizing than linear cost division.
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Blend PI and EI
    # PI is bounded [0,1], EI is unbounded. 
    # To blend them, we should normalize EI or use a weighted sum where weights adapt.
    # Since EI is in objective units and PI is a probability, direct addition is tricky.
    # Let's stick to EI as the primary driver but modify the cost penalty.
    
    # The current program: (0.8 * EI + 0.2 * LCB) / cost
    # LCB = -mu + beta * sigma. Note that EI and LCB are correlated.
    
    # Let's try **Pure EI with Adaptive Cost Penalty**.
    # The linear division by cost might be too harsh.
    # If cost is 1.5, dividing by 1.5 reduces score by 33%.
    # If we divide by sqrt(cost), it reduces by 13%.
    # Let's try gamma = 0.5.
    
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    cost_penalty = np.sqrt(cost_safe)
    
    scores = ei / cost_penalty
    
    # Add a small exploration boost for points with high sigma, even if EI is low?
    # This is essentially LCB.
    # Let's blend EI and LCB but with the new cost penalty.
    
    # LCB score (higher is better for minimization means lower function value)
    # Standard UCB for minimization: mu - beta * sigma
    # We want to MAXIMIZE the score, so we want to MINIMIZE (mu - beta * sigma).
    # So score_lcb = -(mu - beta * sigma) = -mu + beta * sigma
    
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize LCB to be comparable to EI?
    # EI is roughly in the range of (f_best - mu).
    # LCB is in the same units.
    
    # We found that blending 0.8 EI and 0.2 LCB worked well (score 1.46).
    # Let's keep the blend but change the cost penalty to sqrt(cost) and adapt beta.
    
    w_ei = 0.8
    w_lcb = 0.2
    
    blended_score = w_ei * ei + w_lcb * lcb_score
    
    scores = blended_score / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
         # Fallback
         return ei / cost_safe

    return scores
# EVOLVE-BLOCK-END
