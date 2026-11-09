import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1e-8)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)

    # z-score for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)

    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)

    # Vectorized Pareto Dominance Check
    # Point i is dominated by point j if EI_j > EI_i and Cost_j < Cost_i
    # We want to find points that are NOT dominated by any other point.
    
    # Broadcasting: 
    # ei_col: shape (N, 1)
    # ei_row: shape (1, N)
    # cost_col: shape (N, 1)
    # cost_row: shape (1, N)
    
    ei_col = ei[:, None]
    ei_row = ei[None, :]
    cost_col = cost[:, None]
    cost_row = cost[None, :]
    
    # j dominates i if ei_row[j, i] > ei_col[i, j] AND cost_row[j, i] < cost_col[i, j]
    # Note: indexing in NumPy is [row, col].
    # Let's define matrix M where M[i, j] is True if j dominates i.
    # M[i, j] = (ei[j] > ei[i]) AND (cost[j] < cost[i])
    
    # ei[j] > ei[i] -> ei_row[:, i] > ei_col[i, :]? 
    # Let's construct the comparison matrix directly.
    # diff_ei = ei_row - ei_col (shape N, N). diff_ei[i, j] = ei[j] - ei[i].
    # diff_cost = cost_row - cost_col (shape N, N). diff_cost[i, j] = cost[j] - cost[i].
    
    # j dominates i if diff_ei[i, j] > 0 AND diff_cost[i, j] < 0.
    
    diff_ei = ei_row - ei_col
    diff_cost = cost_row - cost_col
    
    dominated_by = (diff_ei > 0) & (diff_cost < 0)
    
    # If any j dominates i, then i is dominated.
    is_dominated = np.any(dominated_by, axis=1)
    
    # Pareto Front: not dominated
    is_front = ~is_dominated
    
    # Score: EI / Cost for front points, very low for others
    # We use a large negative number for non-front points to ensure they are never selected
    # unless all points are non-front (which shouldn't happen, as at least one point is on the front)
    
    scores = np.where(is_front, ei / cost, -1e6)
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)

    return scores
