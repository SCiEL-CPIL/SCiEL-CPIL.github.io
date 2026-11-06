"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin from 150s

def _softmin(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick. Returns a smooth approximation to min."""
    if len(values) <= 1:
        return values.copy() if len(values) == 1 else np.array([0.0])
    # Shift for numerical stability
    shifted = values / temperature
    max_val = np.max(shifted)
    # softmin = -temperature * log(sum(exp(-shifted))) + temperature * max_val
    # More numerically stable:
    neg_shifted = -shifted
    neg_max = np.max(neg_shifted)
    log_sum = neg_max + np.log(np.sum(np.exp(neg_shifted - neg_max)))
    return -temperature * log_sum

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _START_TIME, _TIME_LIMIT
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 1:
        return int(feasible_nodes[0])
    
    # Compute distances
    dist_from_current = distance_matrix[current_node, feasible_nodes].astype(np.float64)
    dist_from_depot = distance_matrix[depot, feasible_nodes].astype(np.float64)
    
    # Compute savings score: maximize dist(depot, next) - dist(current, next)
    savings_score = dist_from_depot - dist_from_current
    
    # Compute regret using soft-min surrogate
    # For each node i, regret_i ≈ softmin_j!=i(dist_from_current[j]) - dist_from_current[i]
    # We compute softmin over all nodes, then adjust by removing self-contribution
    # Using a temperature parameter for smoothness
    temperature = 2.0  # Controls smoothness of the soft-min
    
    # Compute soft-min of dist_from_current excluding each node i
    # Efficient approach: for each i, we want softmin of dist_from_current with i removed
    # We can approximate by computing softmin of all, then correcting
    # For better accuracy, use the fact that softmin is dominated by the minimum
    # A cleaner approach: compute pairwise via broadcasting for n <= ~50
    if n_feasible <= 60:
        # Create matrix where row i contains dist_from_current with element i masked
        dist_matrix = np.tile(dist_from_current, (n_feasible, 1))
        # Mask out diagonal (self)
        diag_idx = np.arange(n_feasible)
        dist_matrix[diag_idx, diag_idx] = np.inf
        
        # Compute soft-min for each row
        # For numerical stability, subtract row max before exp
        row_max = np.max(dist_matrix, axis=1)
        # Handle case where all are inf (shouldn't happen with n>=2)
        row_max_safe = np.where(np.isinf(row_max), 0.0, row_max)
        
        exp_terms = np.exp(-(dist_matrix - row_max_safe[:, np.newaxis]) / temperature)
        # Set inf entries to 0 in exp
        exp_terms[~np.isfinite(dist_matrix)] = 0.0
        
        log_sums = np.log(np.sum(exp_terms, axis=1) + 1e-300)
        softmin_excl = row_max_safe - temperature * log_sums
    else:
        # Fallback for larger n: use the full softmin as approximation
        softmin_all = _softmin(dist_from_current, temperature)
        softmin_excl = np.full(n_feasible, softmin_all)
    
    # Regret: positive means node i is closer to current than the soft-min of alternatives
    regret = softmin_excl - dist_from_current
    
    # Normalize regret to [0, 1] for combination
    max_regret = np.max(regret)
    min_regret = np.min(regret)
    if max_regret > min_regret:
        regret_normalized = (regret - min_regret) / (max_regret - min_regret)
    else:
        regret_normalized = np.zeros(n_feasible)
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings_score)
    min_savings = np.min(savings_score)
    if max_savings > min_savings:
        savings_normalized = (savings_score - min_savings) / (max_savings - min_savings)
    else:
        savings_normalized = np.zeros(n_feasible)
    
    # Dynamic weighting: when capacity is low, weight regret more heavily
    total_capacity = 50
    capacity_ratio = rest_capacity / total_capacity
    
    # When capacity is low, prioritize high-regret nodes more
    if capacity_ratio < 0.3:
        regret_weight = 0.45 + 0.25 * (0.3 - capacity_ratio) / 0.3  # 0.45 to 0.70
        savings_weight = 1.0 - regret_weight
    else:
        regret_weight = 0.30
        savings_weight = 0.70
    
    # Combined score
    combined_score = savings_weight * savings_normalized + regret_weight * regret_normalized
    
    # Add a small penalty for very long distances to avoid extreme detours
    max_dist = np.max(dist_from_current) if len(dist_from_current) > 0 else 1.0
    if max_dist > 0:
        dist_penalty = 0.03 * (dist_from_current / max_dist)
        combined_score = combined_score - dist_penalty
    
    # We want to maximize the combined score
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
