"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    start_time = time.time()
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure we have the right shape
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        total_prizes = prizes.sum(axis=0)
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        total_prizes = prizes
    else:
        total_prizes = prizes.reshape(-1)
        if total_prizes.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_prizes.shape[0])
            tmp[:n] = total_prizes[:n]
            total_prizes = tmp
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        total_weights = weights.sum(axis=0)
    elif weights.ndim == 1 and weights.shape[0] == 100:
        total_weights = weights
    else:
        total_weights = weights.reshape(-1)
        if total_weights.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_weights.shape[0])
            tmp[:n] = total_weights[:n]
            total_weights = tmp
    
    # Base ratio heuristic
    ratio = total_prizes / (total_weights + 1e-9)
    
    # Bottleneck-aware adjustment: penalize items that heavily load a single constraint
    # This helps avoid items that block other high-value items
    if weights.ndim == 2 and weights.shape[0] == 5:
        # For each item, compute the maximum weight in any constraint (bottleneck)
        max_weight_per_item = weights.max(axis=0)
        # Soft-min of (capacity - weight) across constraints: how much slack remains
        # Use log-sum-exp for smooth approximation
        capacity = 25.0
        slacks = capacity - weights  # shape (5, 100)
        # Soft-min via log-sum-exp of negatives
        max_neg_slacks = np.max(-slacks, axis=0)  # = -min(slacks)
        # Soft-min = -logsumexp(-slacks)
        # For numerical stability: soft_min = -max_neg - log(sum(exp(-slacks - max_neg)))
        exp_terms = np.exp(-slacks - max_neg_slacks[None, :])
        log_sum = np.log(np.sum(exp_terms, axis=0))
        soft_min_slack = -max_neg_slacks - log_sum
        
        # Bottleneck penalty: items with low soft-min slack are more constraining
        # Normalize slack to [0, 1] range
        min_slack = soft_min_slack.min()
        max_slack = soft_min_slack.max()
        if max_slack > min_slack + 1e-9:
            normalized_slack = (soft_min_slack - min_slack) / (max_slack - min_slack)
        else:
            normalized_slack = np.ones_like(soft_min_slack)
        
        # Penalty factor: reduce score for items with low slack (high bottleneck)
        # Use a smooth penalty that doesn't go to zero
        bottleneck_factor = 0.5 + 0.5 * normalized_slack  # in [0.5, 1.0]
        
        # Also consider the ratio of max weight to total weight (concentration)
        weight_concentration = max_weight_per_item / (total_weights + 1e-9)
        # Items with high concentration in one constraint are more likely to block
        concentration_penalty = 1.0 / (1.0 + weight_concentration)
        
        # Combine: ratio * bottleneck_factor * concentration_penalty
        scores = ratio * bottleneck_factor * concentration_penalty
        
        # Co-occurrence penalty via greedy packing analysis
        # Simulate a greedy packing to identify which items conflict with top items
        # and penalize items that share high weight correlation with selected items
        
        # Perform greedy packing: select items by ratio until no more fit
        remaining_cap = np.full(5, capacity)
        selected_items = []
        remaining_indices = np.arange(100)
        
        while len(remaining_indices) > 0:
            # Find best ratio item that fits
            best_idx = -1
            best_ratio = -1
            for idx in remaining_indices:
                if np.all(weights[:, idx] <= remaining_cap):
                    if ratio[idx] > best_ratio:
                        best_ratio = ratio[idx]
                        best_idx = idx
            if best_idx == -1:
                break
            selected_items.append(best_idx)
            remaining_cap -= weights[:, best_idx]
            remaining_indices = remaining_indices[remaining_indices != best_idx]
        
        # Compute co-occurrence penalty based on weight correlation with selected items
        if len(selected_items) > 0:
            # For each item, compute average weight correlation with selected items
            # High correlation means likely to compete for same resources
            weight_matrix = weights.T  # shape (100, 5)
            selected_weights = weight_matrix[selected_items]  # shape (n_selected, 5)
            
            # Compute correlation-like measure: dot product normalized
            # For each item i, compute similarity to each selected item j
            # Use normalized dot product as proxy for resource conflict
            norms_i = np.linalg.norm(weight_matrix, axis=1) + 1e-9
            norms_j = np.linalg.norm(selected_weights, axis=1) + 1e-9
            
            # Similarity matrix: (100, n_selected)
            sim_matrix = (weight_matrix @ selected_weights.T) / (np.outer(norms_i, norms_j))
            
            # Average similarity to selected items
            avg_similarity = sim_matrix.mean(axis=1)
            
            # Normalize similarity to [0, 1]
            min_sim = avg_similarity.min()
            max_sim = avg_similarity.max()
            if max_sim > min_sim + 1e-9:
                normalized_sim = (avg_similarity - min_sim) / (max_sim - min_sim)
            else:
                normalized_sim = np.ones_like(avg_similarity)
            
            # Penalty: items with high similarity to selected items get reduced score
            # This discourages choosing items that conflict with already-selected high-value items
            cooccurrence_penalty = 1.0 / (1.0 + 2.0 * normalized_sim)
            
            # Apply co-occurrence penalty
            scores = scores * cooccurrence_penalty
    else:
        scores = ratio
    
    # Sharpen with power to create stronger differentiation
    scores = scores ** 2.0
    scores = np.where(scores > 0, scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 320:
        pass  # Return whatever we have
    
    return scores
# EVOLVE-BLOCK-END
