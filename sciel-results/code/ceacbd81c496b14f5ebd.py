# EVOLVE-BLOCK-START
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Find indices of items that fit
    feasible_indices = [i for i in range(n) if remaining_weights[i] <= remaining_capacity]
    
    if not feasible_indices:
        return -1

    # Pre-sort indices by density (value/weight) in descending order
    # This allows us to compute LP relaxation values faster by just iterating
    # and skipping the chosen item.
    densities = [0.0] * n
    for i in range(n):
        if remaining_weights[i] > 0:
            densities[i] = remaining_values[i] / remaining_weights[i]
        else:
            densities[i] = float('inf')
            
    sorted_indices = sorted(range(n), key=lambda x: densities[x], reverse=True)

    best_index = -1
    best_score = -1.0
    best_density = -1.0
    best_value = -1.0
    best_weight = float('inf')

    # Helper to compute LP relaxation value for remaining items (excluding 'excluded_idx')
    # using the pre-sorted order.
    def compute_lp_value_excluding(cap, excluded_idx):
        if cap <= 0:
            return 0.0
        
        total_val = 0.0
        rem_cap = cap
        
        for idx in sorted_indices:
            if idx == excluded_idx:
                continue
            w = remaining_weights[idx]
            v = remaining_values[idx]
            if w == 0:
                total_val += v
            elif rem_cap >= w:
                total_val += v
                rem_cap -= w
            else:
                total_val += v * (rem_cap / w)
                rem_cap = 0
                break
                
        return total_val

    for i in feasible_indices:
        new_cap = remaining_capacity - remaining_weights[i]
        
        # Calculate LP value of the rest, excluding item i
        lp_val = compute_lp_value_excluding(new_cap, i)
        score = remaining_values[i] + lp_val
        
        # Calculate density for tie-breaking
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        d_i = v_i / w_i if w_i > 0 else float('inf')
        
        # Check for improvement
        epsilon = 1e-9
        if score > best_score + epsilon:
            best_score = score
            best_index = i
            best_density = d_i
            best_value = v_i
            best_weight = w_i
        elif abs(score - best_score) <= epsilon:
            # Tie-breaking logic
            # 1. Higher density is preferred
            if d_i > best_density + 1e-9:
                best_index = i
                best_density = d_i
                best_value = v_i
                best_weight = w_i
            elif abs(d_i - best_density) <= 1e-9:
                # 2. Higher value is preferred
                if v_i > best_value + 1e-9:
                    best_index = i
                    best_density = d_i
                    best_value = v_i
                    best_weight = w_i
                elif abs(v_i - best_value) <= 1e-9:
                    # 3. Lower weight is preferred
                    if w_i < best_weight - 1e-9:
                        best_index = i
                        best_density = d_i
                        best_value = v_i
                        best_weight = w_i

    return best_index
# EVOLVE-BLOCK-END
