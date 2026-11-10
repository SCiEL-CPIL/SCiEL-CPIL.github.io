import math

def construct():
    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    D = 1
    for i in range(1, 31):
        D = lcm(D, i)
        
    # Weights w_i = D / i
    weights = [D // i for i in range(1, 31)]
    
    # Beam Search
    # State: (abs_sum, sum, coeffs_list)
    # We want to minimize abs_sum.
    # If abs_sum is 0, we need to handle it carefully at the end.
    
    beam_width = 500
    # Initial state
    beam = [(0, 0, [])] # (abs_val, val, coeffs)
    
    for i in range(30):
        w = weights[i]
        new_beam = []
        for abs_val, val, coeffs in beam:
            # Try -1, 0, 1
            for c in [-1, 0, 1]:
                new_val = val + c * w
                new_abs = abs(new_val)
                new_coeffs = coeffs + [c]
                new_beam.append((new_abs, new_val, new_coeffs))
        
        # Sort by new_abs, then maybe by something else to keep diversity?
        # Just sorting by abs is fine. If ties, we might lose diversity.
        # Let's sort by abs, then by the number of non-zeros (prefer more non-zeros? 
        # Actually, for finding 0, it doesn't matter. For finding min non-zero, 
        # having more flexibility (more 0s) might be better? 
        # Let's just sort by abs.
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
    # Find the best candidate
    best_candidate = None
    min_abs = float('inf')
    
    for abs_val, val, coeffs in beam:
        if val == 0:
            # If we found a zero sum, we can construct a minimal non-zero sum.
            # The smallest change is 1/max_index where c_max_index is 0? 
            # Or 2/max_index where c_max_index is +/-1?
            # Let's check the last few indices.
            # We can just pick the smallest possible non-zero value we can generate from this zero state.
            # min_change = min(1/j for j in 1..30 if coeffs[j-1]==0) 
            #             or min(2/j for j in 1..30 if coeffs[j-1]!=0)
            # Actually, we can just return a valid non-zero sum derived from this.
            # Let's find the smallest unit change.
            min_change_val = float('inf')
            min_change_idx = -1
            for j in range(30):
                idx = j + 1
                if coeffs[j] == 0:
                    # Change 0 to 1: sum becomes 1/idx -> num = D/idx
                    change_num = D // idx
                    if change_num < min_change_val:
                        min_change_val = change_num
                        min_change_idx = j
                else:
                    # Change 1 to -1 or -1 to 1: sum becomes +/- 2/idx -> num = 2*D/idx
                    change_num = 2 * (D // idx)
                    if change_num < min_change_val:
                        min_change_val = change_num
                        min_change_idx = j
            
            if min_change_idx != -1:
                # Apply the change
                new_coeffs = coeffs[:]
                if new_coeffs[min_change_idx] == 0:
                    new_coeffs[min_change_idx] = 1
                else:
                    # Flip sign
                    new_coeffs[min_change_idx] = -new_coeffs[min_change_idx]
                
                # Calculate final sum
                final_val = 0
                for c, wt in zip(new_coeffs, weights):
                    final_val += c * wt
                final_abs = abs(final_val)
                
                if final_abs < min_abs:
                    min_abs = final_abs
                    best_candidate = new_coeffs
        else:
            if abs_val < min_abs:
                min_abs = abs_val
                best_candidate = coeffs
                
    if best_candidate is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return best_candidate
