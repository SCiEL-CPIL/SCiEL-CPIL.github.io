import random
import numpy as np

def construct():
    N = 256
    rng = random.Random(171)
    s = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=int)
    
    # Precompute APs grouped by step d?
    # Or just iterate d, a, L in get_worst.
    
    def get_worst(s):
        max_abs = 0
        worst_ap = None
        worst_sum = 0
        
        for d in range(1, N + 1):
            # Compute prefix sums for each residue class
            # prefix[r] will be a list of cumulative sums
            # We only need to check APs that start at 'a'
            
            # For each residue r in 0..d-1
            for r in range(d):
                # Get indices: r, r+d, r+2d...
                # We can compute prefix sums on the fly
                # Or extract the subarray
                # sub = s[r::d]
                # prefix = np.cumsum(sub)
                # Then for each start a = r + k*d, and length L:
                # sum = prefix[k+L-1] - prefix[k-1] (if k>0) else prefix[L-1]
                
                sub = s[r::d]
                if len(sub) == 0:
                    continue
                prefix = np.cumsum(sub)
                
                # Iterate over possible starts k (index in sub)
                # a = r + k*d
                # Max L such that k + L - 1 < len(sub)
                # L < len(sub) - k
                # L <= len(sub) - k - 1 + 1 = len(sub) - k
                for k in range(len(sub)):
                    max_L = len(sub) - k
                    # For each L from 1 to max_L
                    # sum = prefix[k+L-1] - (prefix[k-1] if k>0 else 0)
                    base = prefix[k-1] if k > 0 else 0
                    # We want to find the L that maximizes abs(sum)
                    # We can just iterate L
                    for L in range(1, max_L + 1):
                        curr_sum = prefix[k + L - 1] - base
                        abs_curr = abs(curr_sum)
                        if abs_curr > max_abs:
                            max_abs = abs_curr
                            worst_ap = (r + k * d, d, L) # a = r + k*d
                            worst_sum = curr_sum
                            
        return max_abs, worst_ap, worst_sum

    current_max, worst_ap, worst_sum = get_worst(s)
    
    iterations = 500
    
    for _ in range(iterations):
        if current_max <= 2:
            break
        a, d, L = worst_ap
        best_idx = -1
        best_new_abs = abs(worst_sum)
        
        # Indices in this AP: a, a+d, ..., a+(L-1)d
        # 0-based
        indices = range(a, a + L*d, d)
        
        for idx in indices:
            new_sum = worst_sum - 2 * s[idx]
            new_abs = abs(new_sum)
            if new_abs < best_new_abs:
                best_new_abs = new_abs
                best_idx = idx
                if best_new_abs == 0:
                    break
        
        if best_idx != -1:
            s[best_idx] *= -1
            current_max, worst_ap, worst_sum = get_worst(s)
        else:
            # Random flip in worst AP
            idx = random.choice(list(indices))
            s[idx] *= -1
            current_max, worst_ap, worst_sum = get_worst(s)
            
    return s.tolist()
