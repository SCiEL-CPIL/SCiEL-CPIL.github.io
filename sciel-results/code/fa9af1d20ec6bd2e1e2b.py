"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 105.0  # safety margin before 120s kill


def _greedy_complete_fast(capacity, weights, values, used_mask=None):
    """
    Complete greedily by value density using a pre-sorted order.
    This is O(n log n) for sorting + O(n) for the fill, much faster than O(n^2) np.delete.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Create a mask of available items (not used)
    if used_mask is None:
        available = np.ones(n, dtype=bool)
    else:
        available = ~used_mask
    
    # Filter to feasible items
    feasible = available & (weights <= capacity)
    if not np.any(feasible):
        return 0.0
    
    idxs = np.where(feasible)[0]
    w = weights[idxs]
    v = values[idxs]
    
    # Sort by density descending
    densities = v / w
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        if w[idx] <= remaining_cap:
            total_value += v[idx]
            remaining_cap -= w[idx]
    
    return total_value


def _polish_swap(chosen_mask, weights, values, capacity, max_iters=100):
    """
    Try to improve a chosen set by swapping one chosen item for one or more unchosen items.
    This is a simple local search: for each chosen item, see if removing it allows adding
    better items.
    Returns improved chosen mask.
    """
    n = len(weights)
    chosen_mask = chosen_mask.copy()
    
    if not np.any(chosen_mask):
        return chosen_mask
    
    for _ in range(max_iters):
        improved = False
        chosen_idxs = np.where(chosen_mask)[0]
        unchosen_idxs = np.where(~chosen_mask)[0]
        
        if len(unchosen_idxs) == 0:
            break
        
        # For each chosen item, check if swapping improves
        for i in chosen_idxs:
            if improved:
                break
            w_out = weights[i]
            v_out = values[i]
            
            # If we remove item i, we free up w_out capacity
            new_cap = capacity - (np.sum(weights[chosen_mask]) - w_out)
            
            # Find best unchosen item that fits in new_cap
            candidates = unchosen_idxs[weights[unchosen_idxs] <= new_cap]
            if len(candidates) == 0:
                continue
            
            # Choose the candidate with highest value
            best_candidate = candidates[np.argmax(values[candidates])]
            
            # Check if this swap improves total value
            if values[best_candidate] > v_out:
                # Perform swap
                chosen_mask[i] = False
                chosen_mask[best_candidate] = True
                improved = True
        
        if not improved:
            break
    
    return chosen_mask


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item to add to the knapsack using lookahead-guided selection with polish.
    Returns index into remaining arrays, or -1 to stop.
    """
    # Check time budget
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check if any item is feasible
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    # For small n, use exhaustive evaluation
    if n <= 20:
        feasible_idxs = np.where(feasible)[0]
        best_score = -1
        best_idx = -1
        
        for idx in feasible_idxs:
            used = np.zeros(n, dtype=bool)
            used[idx] = True
            greedy_val = _greedy_complete_fast(
                remaining_capacity - remaining_weights[idx],
                remaining_weights,
                remaining_values,
                used_mask=used
            )
            total_val = remaining_values[idx] + greedy_val
            
            if total_val > best_score:
                best_score = total_val
                best_idx = idx
        
        return int(best_idx)
    
    else:
        # For larger instances, use top-k candidates by density
        feasible_idxs = np.where(feasible)[0]
        densities = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
        
        # Take top min(10, n) candidates by density
        k = min(10, len(feasible_idxs))
        top_k_local = np.argsort(-densities)[:k]
        top_k_global = feasible_idxs[top_k_local]
        
        best_score = -1
        best_idx = -1
        best_chosen = None
        
        for idx in top_k_global:
            used = np.zeros(n, dtype=bool)
            used[idx] = True
            greedy_val = _greedy_complete_fast(
                remaining_capacity - remaining_weights[idx],
                remaining_weights,
                remaining_values,
                used_mask=used
            )
            total_val = remaining_values[idx] + greedy_val
            
            if total_val > best_score:
                best_score = total_val
                best_idx = idx
                # Store the chosen mask for potential polish
                best_chosen = used.copy()
        
        # Try to polish the best solution found
        if best_chosen is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
            polished = _polish_swap(best_chosen, remaining_weights, remaining_values, remaining_capacity)
            
            # Check if polish improved the solution
            if np.sum(polished) > 0:
                polished_value = np.sum(remaining_values[polished])
                if polished_value > best_score:
                    # Find the first item in the polished solution to return
                    # We want to return an index that leads to the polished solution
                    # Since we're building incrementally, we just return the best candidate
                    # that was part of the polished solution
                    polished_idxs = np.where(polished)[0]
                    # Prefer items from our top-k candidates if they're in the polished set
                    for idx in top_k_global:
                        if polished[idx]:
                            return int(idx)
                    # Otherwise return the first polished item
                    return int(polished_idxs[0])
        
        return int(best_idx)
# EVOLVE-BLOCK-END
