import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to only items that fit
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    feasible_count = len(feasible_indices)
    
    if feasible_count == 0:
        return -1
    
    # If only one feasible item, pick it
    if feasible_count == 1:
        return int(feasible_indices[0])
    
    # Calculate densities for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0)
    
    # Sort by density descending for greedy fill
    sorted_order = np.argsort(densities)[::-1]
    
    def greedy_fill(capacity, exclude_indices):
        """Fill knapsack greedily by density, excluding certain indices."""
        total_value = 0.0
        cap = capacity
        exclude_set = set(exclude_indices)
        for idx in sorted_order:
            if idx in exclude_set:
                continue
            if weights[idx] <= cap:
                total_value += values[idx]
                cap -= weights[idx]
        return total_value
    
    # For each feasible candidate, simulate: pick it, then greedily fill the rest
    # Use 2-step lookahead: pick candidate, then pick best second item, then fill
    best_index = int(feasible_indices[0])
    best_total_value = -1
    
    # Limit candidates to top ~30 by density to keep it fast, but ensure we check all feasible
    # Actually with n=100, checking all feasible is fine
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        cap_after = remaining_capacity - w_i
        
        if cap_after <= 0:
            total = v_i
        else:
            # 2-step lookahead: try each second item, then greedy fill
            # To speed up, only consider top candidates for second pick
            # Get items that fit after picking i
            second_feasible = weights <= cap_after
            second_feasible[i] = False  # can't pick same item
            
            if not np.any(second_feasible):
                total = v_i
            else:
                # Try top few second picks by density
                second_candidates = np.where(second_feasible)[0]
                # Sort second candidates by density
                second_densities = densities[second_candidates]
                second_sorted = second_candidates[np.argsort(second_densities)[::-1]]
                
                # Only try top 10 second picks for speed
                max_second_tries = min(10, len(second_sorted))
                best_second_total = -1
                
                for j in second_sorted[:max_second_tries]:
                    cap_after2 = cap_after - weights[j]
                    if cap_after2 <= 0:
                        second_total = v_i + values[j]
                    else:
                        # Greedy fill the rest
                        third_feasible = weights <= cap_after2
                        third_feasible[i] = False
                        third_feasible[j] = False
                        
                        if not np.any(third_feasible):
                            second_total = v_i + values[j]
                        else:
                            # Greedy fill
                            fill_value = 0.0
                            cap_fill = cap_after2
                            exclude = {i, j}
                            for idx in sorted_order:
                                if idx in exclude:
                                    continue
                                if weights[idx] <= cap_fill:
                                    fill_value += values[idx]
                                    cap_fill -= weights[idx]
                            second_total = v_i + values[j] + fill_value
                    
                    if second_total > best_second_total:
                        best_second_total = second_total
                
                total = best_second_total
        
        if total > best_total_value:
            best_total_value = total
            best_index = int(i)
    
    return best_index
