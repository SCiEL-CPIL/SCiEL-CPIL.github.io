import numpy as np
import time

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Ensure both have the same number of rows
    if prizes.shape[0] == 1 and weights.shape[0] > 1:
        prizes = np.repeat(prizes, weights.shape[0], axis=0)
    elif weights.shape[0] == 1 and prizes.shape[0] > 1:
        weights = np.repeat(weights, prizes.shape[0], axis=0)
    
    num_constraints, num_items = prizes.shape
    
    eps = 1e-9
    
    # Compute per-constraint efficiency: prize[c, i] / weight[c, i]
    safe_weights = np.where(weights < eps, eps, weights)
    ratios = prizes / safe_weights  # shape (num_constraints, num_items)
    
    # Normalize weights per constraint to capture relative tightness
    # For each constraint c, compute the mean weight across items
    mean_weight_per_constraint = np.mean(weights, axis=1, keepdims=True)  # shape (num_constraints, 1)
    mean_weight_per_constraint = np.where(mean_weight_per_constraint < eps, eps, mean_weight_per_constraint)
    
    # Normalize: w_norm[c, i] = weights[c, i] / mean_weight_per_constraint[c]
    # This makes the average weight per constraint = 1
    normalized_weights = weights / mean_weight_per_constraint  # shape (num_constraints, num_items)
    
    # Compute normalized efficiency: prize[c, i] / normalized_weight[c, i]
    # This is equivalent to prize[c, i] * mean_weight_per_constraint[c] / weights[c, i]
    safe_normalized_weights = np.where(normalized_weights < eps, eps, normalized_weights)
    normalized_ratios = prizes / safe_normalized_weights  # shape (num_constraints, num_items)
    
    # Geometric mean across constraints
    normalized_ratios = np.maximum(normalized_ratios, eps)
    log_ratios = np.log(normalized_ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)  # shape (num_items,)
    geo_mean_efficiency = np.exp(mean_log_ratio)  # shape (num_items,)
    
    # Ensure positive
    heuristic = np.maximum(geo_mean_efficiency, eps)
    
    return heuristic
