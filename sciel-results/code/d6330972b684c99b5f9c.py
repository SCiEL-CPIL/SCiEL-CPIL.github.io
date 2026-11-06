import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Weights
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums_and_reachables(weights):
        """
        Generates the set of all possible sums and the history of reachable sets
        for backtracking.
        Returns:
        - sums_list: sorted list of all possible sums
        - reachables: list of sets, where reachables[i] contains sums reachable 
                      using weights[i:]
        """
        reachables = []
        curr = {0}
        reachables.append(curr)
        
        for wi in reversed(weights):
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            reachables.append(curr)
            
        # Reverse to get reachables[0] as full set, reachables[k] as sums from k to end
        # Wait, the loop above builds from end to start.
        # reachables[0] = {0} (empty set of weights)
        # reachables[1] = sums using last weight
        # ...
        # reachables[len(weights)] = sums using all weights
        
        # We want reachables[i] to be the set of sums using weights[i:]
        # So if we iterate weights in reverse, and prepend to reachables?
        # Let's re-structure for clarity.
        
        pass

    def generate_sums_and_reachables_v2(weights):
        n_w = len(weights)
        # reachables[i] will store the set of sums achievable using weights[i..n_w-1]
        reachables = [None] * (n_w + 1)
        reachables[n_w] = {0}
        
        for i in range(n_w - 1, -1, -1):
            wi = weights[i]
            prev_set = reachables[i+1]
            curr_set = set()
            for s in prev_set:
                curr_set.add(s - wi)
                curr_set.add(s)
                curr_set.add(s + wi)
            reachables[i] = curr_set
            
        # The full set of sums is reachables[0]
        sums_list = sorted(reachables[0])
        return sums_list, reachables

    def backtrack_coeffs(weights, reachables, target_sum):
        """
        Finds coefficients for weights that sum to target_sum.
        weights[i] corresponds to reachables[i].
        """
        coeffs = [0] * len(weights)
        current_sum = target_sum
        
        for i in range(len(weights)):
            wi = weights[i]
            # We are at index i. We have accumulated 'current_sum' from previous choices?
            # No, 'current_sum' is the remaining sum needed from weights[i:]
            # If we choose c_i = 1, we need current_sum - wi from weights[i+1:]
            # If we choose c_i = 0, we need current_sum from weights[i+1:]
            # If we choose c_i = -1, we need current_sum + wi from weights[i+1:]
            
            if (current_sum - wi) in reachables[i+1]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in reachables[i+1]:
                coeffs[i] = 0
            elif (current_sum + wi) in reachables[i+1]:
                coeffs[i] = -1
                current_sum += wi
            else:
                # Should not happen if target is reachable
                coeffs[i] = 0
                # Fallback logic if needed, but target is from the set so it must be reachable
                # For safety, break or return None
                return None
                
        return coeffs

    # Generate for first half
    sums1, reach1 = generate_sums_and_reachables_v2(w1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    # Generate for second half
    sums2, reach2 = generate_sums_and_reachables_v2(w2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # sums1 is already sorted
    
    best_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    # Iterate through sums2
    # To speed up, we can iterate through sums2 and binary search in sums1
    # sums2 is not sorted in generate_sums_and_reachables_v2? 
    # reachables[0] is a set. We sorted it for sums1. 
    # We should sort sums2 as well? Or just iterate the set?
    # Iterating a set is fine.
    
    for s2 in reach2[0]:
        target = -s2
        idx = bisect.bisect_left(sums1, target)
        
        # Check neighbors
        # We want min |s1 + s2| != 0
        candidates = set()
        for check_idx in [idx, idx - 1, idx + 1, idx - 2, idx + 2]:
            if 0 <= check_idx < len(sums1):
                candidates.add(sums1[check_idx])
                
        for s1 in candidates:
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_val:
                    best_val = abs_total
                    best_s1 = s1
                    best_s2 = s2

    if time.time() > deadline or best_s1 is None:
        return [0] * 28 + [1, -1]
        
    # Backtrack to get coefficients
    coeffs1 = backtrack_coeffs(w1, reach1, best_s1)
    coeffs2 = backtrack_coeffs(w2, reach2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    result = coeffs1 + coeffs2
    
    # Verify
    total = 0
    for i, c in enumerate(result):
        if c != 0:
            total += c * (L // (i + 1))
            
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result
