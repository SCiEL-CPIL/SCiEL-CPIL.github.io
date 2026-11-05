import math
import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Denominators are 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2 to 48
    L = 1
    for d in denominators:
        L = math.lcm(L, d)
        
    # Weights are L / d
    weights = [L // d for d in denominators]
    
    # Split indices into two halves
    mid = n // 2
    left_weights = weights[:mid]
    right_weights = weights[mid:]
    
    left_indices = list(range(mid))
    right_indices = list(range(mid, n))
    
    # Function to generate all subset sums and masks
    def generate_subsets(weights_list):
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            # We need to add w to all existing sums
            # To avoid modifying list while iterating, we iterate over current length
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                sums.append((s + w, m | (1 << i)))
        return sums

    # Generate subsets for left half
    left_subsets = generate_subsets(left_weights)
    # Sort by sum value
    left_subsets.sort(key=lambda x: x[0])
    
    # Extract values for binary search
    left_vals = [s[0] for s in left_subsets]
    left_masks = [s[1] for s in left_subsets]
    
    # Generate subsets for right half and find best combination
    best_sum = -1
    best_left_mask = 0
    best_right_mask = 0
    
    right_subsets = generate_subsets(right_weights)
    
    for s_r, m_r in right_subsets:
        # We need s_l + s_r < L  => s_l <= L - 1 - s_r
        target = L - 1 - s_r
        if target < 0:
            continue
            
        # Find the largest s_l <= target
        # bisect_right returns insertion point to the right of existing entries
        # So idx - 1 is the largest index with val <= target
        idx = bisect.bisect_right(left_vals, target)
        if idx > 0:
            s_l = left_vals[idx - 1]
            total = s_l + s_r
            if total > best_sum:
                best_sum = total
                best_left_mask = left_masks[idx - 1]
                best_right_mask = m_r
                
        # Check time
        if time.time() > deadline:
            break

    # If no valid sum found (shouldn't happen as 0 is always valid), return empty? 
    # But problem says nonempty. 
    # Sum 0 is valid (empty set), but we need nonempty.
    # However, we want to minimize deficit, so we want max sum < L.
    # The empty set has sum 0. Deficit 1.
    # There are definitely subsets with sum > 0 and < L.
    
    # Reconstruct indices
    selected_indices = []
    
    # Left mask bits correspond to left_indices
    for i in range(len(left_indices)):
        if best_left_mask & (1 << i):
            selected_indices.append(left_indices[i])
            
    # Right mask bits correspond to right_indices
    for i in range(len(right_indices)):
        if best_right_mask & (1 << i):
            selected_indices.append(right_indices[i])
            
    # If selected_indices is empty, we must pick at least one.
    # But best_sum will be > 0 because 1/2 < 1.
    # So selected_indices should not be empty.
    
    return selected_indices
