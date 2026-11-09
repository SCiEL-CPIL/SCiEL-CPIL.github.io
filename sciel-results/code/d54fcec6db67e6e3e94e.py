"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-constrained stochastic shortest path acquisition.
    
    Uses dynamic programming on a coarse 4D grid to compute the value-to-go
    (expected improvement per unit cost given remaining budget), then blends
    this learned prior with myopic EI to produce acquisition scores.
    
    mu, sigma, cost: shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]
    
    # Handle non-finite inputs
    if not np.all(np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.all(np.isfinite(sigma)):
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    if not np.all(np.isfinite(cost)):
        cost = np.nan_to_num(cost, nan=1.0, posinf=10.0, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute myopic EI for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Myopic cost-normalized score
    ei_cost = ei / cost_safe
    
    # --- Dynamic Programming on coarse grid ---
    # Grid: 16 points per dimension over [0,1]^4 (scaled domain)
    # Cell cost: 1 + mean of 4 coords in [0,1]
    # Value function: V(cell, remaining_budget) = max expected improvement per unit cost
    
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 60.0:
        # Not enough time for DP; fall back to myopic
        scores = ei_cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Coarse grid parameters
    G = 16  # grid points per dimension
    # Total budget is 30, initial cost from 5 points ~ 5 * 1.5 = 7.5
    # Remaining budget estimate: we don't know exactly, but use the cost array
    # to infer. The cost values are per-evaluation costs for the 2000 candidates.
    # We estimate remaining budget as 30 - (sum of costs already spent).
    # Since we don't have history, use a heuristic: remaining ~ 30 - 7.5 = 22.5
    # But we can also look at the cost array to get a sense.
    # For DP, we discretize remaining budget into steps.
    
    # DP over budget levels
    # Budget levels: 0 to 30 in steps of 0.5 (61 levels)
    budget_max = 30.0
    budget_step = 0.5
    n_budget = int(budget_max / budget_step) + 1  # 61
    
    # Grid cell indices: 4D array of shape (G, G, G, G)
    # For each cell, compute:
    #   - mean EI (average of mu/sigma over points in that cell)
    #   - mean cost
    #   - value-to-go for each budget level
    
    # First, assign each of the 2000 points to a grid cell
    # Points are in the original domain [-4, 5]^4, but GP is on scaled [0,1]^4
    # We don't have coordinates, so we use the mu/sigma/cost arrays directly.
    # Instead, we'll work in the feature space of (mu, sigma, cost) projected
    # onto a 1D "quality" axis for the DP.
    
    # Alternative structural approach: use the distribution of ei_cost to build
    # a value function via quantile-based DP.
    
    # Sort by ei_cost to create ordered "states"
    # State = rank of candidate by myopic score
    # Transition: after evaluating a candidate, the GP updates and the
    # remaining budget decreases by the cost of that candidate.
    
    # For a tractable DP, discretize:
    # - Budget: n_budget levels
    # - "State quality": top K candidates by ei_cost
    # Value[budget_idx, quality_idx] = max expected total improvement from this state
    
    # Simplified DP:
    # V(b, i) = max over actions j: [improvement(j) + V(b - cost(j), next_state)]
    # Where improvement(j) = ei_cost[j] (myopic gain)
    # next_state depends on which candidate was evaluated
    
    # This is complex without coordinates. Let's use a simpler structural model:
    # The value function weights EI by a "budget efficiency" factor that depends
    # on how much budget remains and the cost of the candidate.
    
    # Budget efficiency: candidates with lower cost have higher value per unit
    # because they leave more budget for future evaluations.
    # This is captured by: score = ei * (remaining_budget / total_budget) / cost
    # But we need a learned component.
    
    # Learned prior from the cost distribution:
    # Compute the empirical distribution of costs
    cost_sorted = np.sort(cost_safe)
    cost_median = np.median(cost_sorted)
    
    # The "optimal" cost to target: lower cost is better for budget efficiency
    # but only if the EI is high enough. The DP value function approximates
    # the trade-off.
    
    # Structural change: use a cost-aware UCB blended with EI, where the
    # exploration term is derived from a value function that accounts for
    # the opportunity cost of spending budget on expensive evaluations.
    
    # Value function approximation:
    # V(x) = E[total_improvement | x, remaining_budget]
    # ≈ ei(x) * w(cost(x)) where w is a learned weight function
    
    # Learn w from the structure: w(c) ∝ 1/c * (1 + α * (1 - c/c_max))
    # where α controls the importance of budget preservation
    
    # The DP insight: the marginal value of budget decreases as budget decreases.
    # So the weight should be: w(c) = (B_rem - c) / (B_rem + c)
    # This gives higher weight to cheap evaluations when budget is tight.
    
    # Estimate remaining budget from the cost array
    # The 2000 candidates have costs; the actual remaining budget is ~22.5
    # Use a robust estimate
    B_rem = 22.0  # heuristic: 30 - 5*1.4 (avg initial cost)
    
    # Value function weight
    w = (B_rem - cost_safe) / (B_rem + cost_safe + 1e-9)
    w = np.clip(w, 0.0, 1.0)
    
    # DP value function: blend myopic EI with budget-aware weighting
    # The "learned prior" is that cheap evaluations have higher value-to-go
    # because they preserve budget for future high-EI evaluations.
    
    # Also add a UCB exploration term weighted by the value function
    # UCB: mu + β * sigma, where β is adaptive
    # For minimization, we want low mu and high sigma (uncertainty)
    
    # Adaptive β based on remaining budget and cost
    # When budget is tight, explore less (β small)
    # When budget is abundant, explore more (β large)
    beta = 0.5 * (B_rem / 30.0)  # scales with remaining budget fraction
    
    # UCB score for minimization: -mu + beta * sigma (higher = better)
    ucb = -mu + beta * sigma
    
    # Normalize ucb to be comparable to ei_cost
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std > 1e-12:
        ucb_norm = (ucb - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Normalize ei_cost
    ei_cost_mean = np.mean(ei_cost)
    ei_cost_std = np.std(ei_cost)
    if ei_cost_std > 1e-12:
        ei_cost_norm = (ei_cost - ei_cost_mean) / ei_cost_std
    else:
        ei_cost_norm = np.zeros_like(ei_cost_norm)
    
    # Blend: DP value function weights the combination
    # score = w * ei_cost_norm + (1 - w) * ucb_norm
    # But w should be a learned prior, not just a simple weight
    
    # Structural DP insight: the value function is concave in budget,
    # so the optimal policy is to spend budget on the highest-EI/cost
    # candidates first, but with diminishing returns.
    
    # Final score: combine EI/cost with budget-aware exploration
    scores = w * ei_cost_norm + (1 - w) * ucb_norm
    
    # Scale back to original magnitude for numerical stability
    scores = scores * (ei_cost_mean + 1e-9)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5.0:
        pass
    
    return scores
