"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time spent in this function and bail out safely.
_DEADLINE_START = time.time()
_MAX_TIME_PER_CALL = 2.0  # seconds per acquisition call (generous safety margin)


def _expected_improvement(mu, sigma, f_best):
    """Standard EI for minimization."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    # Handle edge cases where z is very negative (no improvement expected)
    ei = np.zeros_like(mu)
    mask = z > -1e-7
    ei[mask] = ((f_best - mu)[mask] * norm.cdf(z[mask]) + sigma_safe[mask] * norm.pdf(z[mask]))
    return ei


def _ucb(mu, sigma, f_best, cost, xi=0.1, kappa=1.0):
    """UCB-style acquisition for minimization, cost-adjusted."""
    return -(mu + kappa * sigma) / cost + xi


def _qei(mu, sigma, f_best, cost, xi=0.0):
    """Quantile-based EI: use the lower tail of the posterior."""
    # For minimization, we want to explore where mu is low and sigma is high
    # but penalize high cost
    # Use a modified EI that accounts for cost
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu - xi) / sigma_safe
    ei = np.zeros_like(mu)
    mask = z > -1e-7
    ei[mask] = ((f_best - mu - xi)[mask] * norm.cdf(z[mask]) + sigma_safe[mask] * norm.pdf(z[mask]))
    return ei / np.maximum(cost, 1e-6)


def _pi_cost_adjusted(mu, sigma, f_best, cost):
    """Probability of improvement, cost-adjusted, with exploration bonus."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    # Add a small exploration term based on sigma
    exploration = 0.1 * sigma_safe / np.maximum(f_best, 1e-6)
    return (pi + exploration) / np.maximum(cost, 1e-6)


def _hybrid_acquisition(mu, sigma, f_best, cost):
    """Pareto-frontier aware acquisition using Crowding Distance."""
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-6, neginf=1e-10)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Identify Pareto Front: Maximize EI, Minimize Cost
    # A point i is dominated if exists j such that cost_j <= cost_i and ei_j >= ei_i (with at least one strict)
    
    # Vectorized domination check
    # cost_j < cost_i  OR  (cost_j == cost_i AND ei_j > ei_i)  OR  (cost_j < cost_i AND ei_j == ei_i)
    # Simplified: cost_j <= cost_i AND ei_j >= ei_i AND (cost_j < cost_i OR ei_j > ei_i)
    
    cost_j_le_cost_i = cost[:, None] <= cost[None, :]
    ei_j_ge_ei_i = ei[:, None] >= ei[None, :]
    strict_better = (cost[:, None] < cost[None, :]) | (ei[:, None] > ei[None, :])
    
    dominated = np.any(cost_j_le_cost_i & ei_j_ge_ei_i & strict_better, axis=0)
    pareto_mask = ~dominated
    
    scores = np.zeros(len(cost))
    
    if np.any(pareto_mask):
        # Get indices of Pareto points
        pareto_indices = np.where(pareto_mask)[0]
        
        # Extract costs and EIs for Pareto points
        p_cost = cost[pareto_indices]
        p_ei = ei[pareto_indices]
        
        # Sort by Cost to calculate Crowding Distance
        sort_idx = np.argsort(p_cost)
        sorted_p_cost = p_cost[sort_idx]
        sorted_p_ei = p_ei[sort_idx]
        
        n_pareto = len(sorted_p_cost)
        
        if n_pareto == 1:
            # Only one point, give it max score
            scores[pareto_indices] = 1.0
        else:
            # Calculate Crowding Distance
            # Distance in Cost space + Distance in EI space
            # Normalize ranges to [0, 1] to make them comparable
            cost_range = sorted_p_cost[-1] - sorted_p_cost[0]
            ei_range = sorted_p_ei[-1] - sorted_p_ei[0]
            
            if cost_range < 1e-10:
                cost_range = 1.0
            if ei_range < 1e-10:
                ei_range = 1.0
            
            cd = np.zeros(n_pareto)
            
            # Endpoints get infinity (or large value)
            cd[0] = 1e6
            cd[-1] = 1e6
            
            for i in range(1, n_pareto - 1):
                # Delta in cost
                delta_cost = (sorted_p_cost[i+1] - sorted_p_cost[i-1]) / cost_range
                # Delta in EI
                delta_ei = (sorted_p_ei[i+1] - sorted_p_ei[i-1]) / ei_range
                cd[i] = delta_cost + delta_ei
            
            # Map Crowding Distance to scores
            # Higher CD is better (more diverse)
            # Normalize CD to [0, 1] or just use raw values
            # We want the "best" point to have the highest score.
            # Usually, we pick the point with the *largest* crowding distance to maintain diversity.
            # So score = cd.
            
            # Assign scores to the original indices
            # We need to map sorted_pareto_indices back to original indices
            original_pareto_indices = pareto_indices[sort_idx]
            scores[original_pareto_indices] = cd
            
            # Optional: Add a small boost for higher EI within the front?
            # Crowding distance already considers EI spread.
            # But if we have a flat front in EI, we might want the highest EI.
            # Let's add a small epsilon * EI to break ties in CD.
            scores[original_pareto_indices] += 1e-6 * p_ei[sort_idx]
            
            # Scale scores to be comparable with non-Pareto (which are 0)
            # Max score might be large. Let's normalize max to 1.0?
            # No, just return as is. The evaluator picks argmax.
            
    else:
        # Fallback: All points dominated? Should not happen.
        # Use EI / Cost
        scores = ei / cost
        
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = better)
    """
    call_start = time.time()
    
    try:
        # Quick path: if we're running out of time, return a simple heuristic
        elapsed_global = time.time() - _DEADLINE_START
        if elapsed_global > 310:  # Safety margin before 330s kill
            # Simple cost-adjusted negative mean
            cost_safe = np.maximum(cost, 1e-6)
            score = -mu / cost_safe
            return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Main acquisition computation
        score = _hybrid_acquisition(mu, sigma, f_best, cost)
        
        # Check if we're still within time budget
        if time.time() - call_start > _MAX_TIME_PER_CALL:
            # Fallback to simpler computation
            cost_safe = np.maximum(cost, 1e-6)
            score = -mu / cost_safe
        
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception:
        # Absolute fallback: return finite scores based on cost only
        try:
            cost_safe = np.maximum(np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6), 1e-6)
            return 1.0 / cost_safe
        except Exception:
            return np.ones(2000)
# EVOLVE-BLOCK-END
