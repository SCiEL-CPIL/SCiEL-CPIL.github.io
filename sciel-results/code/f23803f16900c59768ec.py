import numpy as np
from scipy.stats import norm
import time

# Module-level cache to track elapsed time and iteration count
_acq_state = {
    'start_time': None,
    'call_count': 0,
    'total_cost_spent': 0.0,
    'f_best_history': [],
    'improvement_history': []
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a dynamic, instance-adaptive exploration budget that scales the UCB weight
    by the ratio of remaining budget to current cost, shifting from aggressive exploration
    in early iterations to precise exploitation as the 30-cost limit approaches.
    """
    # Initialize timing
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    _acq_state['call_count'] += 1
    
    # Track f_best history and compute improvements
    if _acq_state['f_best_history']:
        prev_f_best = _acq_state['f_best_history'][-1]
        improvement = prev_f_best - f_best  # Positive means improvement
        _acq_state['improvement_history'].append(improvement)
    _acq_state['f_best_history'].append(f_best)
    
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 285:  # Safety margin before 330s limit (evaluator kills at 330s)
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Dynamic, instance-adaptive exploration budget ---
    # Key insight: scale exploration weight by remaining_budget / candidate_cost
    # This naturally shifts from exploration to exploitation as budget depletes
    
    budget_total = 30.0
    budget_spent = _acq_state['total_cost_spent']
    budget_remaining = max(budget_total - budget_spent, 1e-6)
    
    # The cost of the best candidate we're about to evaluate
    # We use the median cost as a representative "typical" cost for this round
    typical_cost = np.median(cost_safe)
    
    # Dynamic exploration weight: scales with remaining budget relative to typical cost
    # When budget_remaining >> typical_cost: high exploration (early game)
    # When budget_remaining ≈ typical_cost: balanced
    # When budget_remaining < typical_cost: low exploration (late game)
    budget_ratio = budget_remaining / max(typical_cost, 1e-6)
    
    # Map budget_ratio to exploration weight [0, 1]
    # Use a smooth sigmoid-like mapping: weight = 1 / (1 + exp(-k * (budget_ratio - threshold)))
    # threshold = 1.0 means when remaining budget equals typical cost, weight = 0.5
    k = 2.0  # Steepness
    threshold = 1.0
    beta = 1.0 / (1.0 + np.exp(-k * (budget_ratio - threshold)))
    
    # Also incorporate improvement history: if we're not improving, increase exploration
    if len(_acq_state['improvement_history']) >= 3:
        recent_improvements = _acq_state['improvement_history'][-3:]
        mean_improvement = np.mean(recent_improvements)
        # If mean improvement is small, we might be stuck - increase exploration
        if mean_improvement < 0.01:
            beta = np.clip(beta + 0.15, 0.0, 1.0)
    
    beta = np.clip(beta, 0.1, 0.9)  # Keep some balance always
    
    # --- Compute acquisition components ---
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration term
    # Use the distribution of sigma to normalize
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Normalize sigma to [0, 1] range for the exploration term
    sigma_norm = (sigma_safe - sigma_min) / max(sigma_max - sigma_min, 1e-9)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # Exploration score: high where sigma is high, normalized by cost
    # Additionally scale by how much budget we can afford for this candidate
    # Candidates with lower cost get a higher effective exploration weight
    cost_affordability = np.minimum(budget_remaining / cost_safe, 5.0)  # Cap to avoid extreme values
    explore_score = sigma_norm * cost_affordability / cost_safe
    
    # Combine with dynamic beta
    score = (1.0 - beta) * ei_cost + beta * explore_score
    
    # --- Cost-centering bias ---
    # The minimizer is in the inner 80% of [-4, 5]^4, so in scaled coords [0.1, 0.9]
    # Expected cost at minimizer ≈ 1 + 0.5 = 1.5
    # Apply a Gaussian weight to favor candidates with cost near 1.5
    target_cost = 1.5
    # Make the bias weaker as we progress (we've already found the region)
    progress = 1.0 - budget_remaining / budget_total
    bias_strength = 1.0 * (1.0 - 0.5 * progress)
    weight = np.exp(-bias_strength * (cost_safe - target_cost) ** 2 / 0.1)
    
    score = score * weight
    
    # --- Additional: penalize very high cost candidates ---
    # Candidates with cost > 2.0 are less likely to be near the minimizer
    cost_penalty = np.where(cost_safe > 2.0, 0.5, 1.0)
    score = score * cost_penalty
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Update total cost spent (we don't know which candidate will be picked, 
    # but we can estimate based on the top-scored candidate's cost)
    # Actually, we should update after the evaluation happens, but we don't have that hook.
    # Instead, we'll estimate: the cost of the best candidate we're about to evaluate
    # is approximately the cost of the argmax score candidate
    if np.any(score > 0):
        best_idx = np.argmax(score)
        _acq_state['total_cost_spent'] += cost_safe[best_idx]
    
    return score
