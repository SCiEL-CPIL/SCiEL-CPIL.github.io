import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -1
    best_index = -1
    
    # We evaluate each feasible item as a candidate for the next pick
    for i in feasible_indices:
        wi = remaining_weights[i]
        vi = remaining_values[i]
        
        # Capacity remaining after picking i
        cap_rem = remaining_capacity - wi
        
        # If capacity is 0 or less, the lookahead value is 0
        if cap_rem <= 0:
            score = vi
        else:
            # Get the other items
            # Create masks for items that are not i
            mask_other = np.ones(n, dtype=bool)
            mask_other[i] = False
            # Also only consider items that are feasible in the original set? 
            # The function contract says "remaining (feasible) arrays". 
            # Usually, the caller passes only items that fit in the CURRENT capacity.
            # However, after removing i, some items might not fit in cap_rem.
            # The fractional knapsack handles this naturally by stopping when capacity is exhausted.
            
            wo = remaining_weights[mask_other]
            vo = remaining_values[mask_other]
            
            if len(wo) == 0:
                frac_val = 0
            else:
                # Calculate Fractional Knapsack Value for cap_rem using wo, vo
                # Sort by density descending
                densities = vo / wo
                sorted_idx = np.argsort(densities)[::-1]
                
                c = cap_rem
                val = 0
                for idx in sorted_idx:
                    w_curr = wo[idx]
                    v_curr = vo[idx]
                    if w_curr <= c:
                        c -= w_curr
                        val += v_curr
                    else:
                        # Fractional part
                        val += v_curr * (c / w_curr)
                        break
                
                frac_val = val
        
        total_score = vi + frac_val
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
            
    return best_index
