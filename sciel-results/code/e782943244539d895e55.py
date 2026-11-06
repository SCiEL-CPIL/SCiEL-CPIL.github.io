import time
import math
import bisect
from fractions import Fraction

def construct():
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    N = len(denominators)
    
    # Calculate LCM of all denominators
    # lcm(a, b) = abs(a * b) // gcd(a, b)
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    L = 1
    for d in denominators:
        L = (L * d) // gcd(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Greedy Baseline
    # Sort indices by weight descending (which is 1/d descending)
    # i.e., smaller denominators first
    indices_sorted = sorted(range(N), key=lambda i: denominators[i])
    
    greedy_indices = []
    greedy_sum = 0
    for i in indices_sorted:
        if greedy_sum + weights[i] < L:
            greedy_indices.append(i)
            greedy_sum += weights[i]
    
    best_sum = greedy_sum
    best_indices = greedy_indices
    best_mask1 = -1
    best_mask2 = -1
    
    # Meet-in-the-Middle with Pruning
    # Split into two halves
    mid = N // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    start_time = time.time()
    time_limit = 50.0 # Safety margin
    
    def generate_pruned_sums(ws):
        # Returns list of (sum, mask)
        sums = [(0, 0)]
        for i, w in enumerate(ws):
            new_sums = []
            for s, m in sums:
                ns = s + w
                if ns < L:
                    new_sums.append((ns, m | (1 << i)))
            sums.extend(new_sums)
        return sums
    
    try:
        # Generate sums for first half
        sums1 = generate_pruned_sums(w1)
        
        if time.time() - start_time > time_limit * 0.5:
            return best_indices
            
        # Generate sums for second half
        sums2 = generate_pruned_sums(w2)
        
        if time.time() - start_time > time_limit:
            return best_indices
        
        # Sort sums1 by sum
        sums1.sort(key=lambda x: x[0])
        
        # Extract sums for binary search / two pointer
        # We will use two-pointer technique
        # sums1 is sorted ascending.
        # Iterate sums2. For each s2, find max s1 such that s1 + s2 < L.
        
        # To make it faster, we can sort sums2 as well and use two pointers?
        # Or just iterate sums2 and binary search sums1.
        # Binary search is O(log N1).
        # N1 is likely small due to pruning.
        
        # Let's check size of sums1
        # If N1 is huge, we might need to optimize.
        
        # For each s2 in sums2:
        # target = L - 1 - s2
        # Find rightmost s1 <= target
        
        # We can pre-extract the sums list for bisect
        s1_vals = [s for s, m in sums1]
        
        current_best = best_sum
        current_best_m1 = best_mask1
        current_best_m2 = best_mask2
        
        for s2, m2 in sums2:
            target = L - 1 - s2
            if target < 0:
                continue
                
            # Binary search in s1_vals
            # We want the largest value <= target
            idx = bisect.bisect_right(s1_vals, target) - 1
            
            if idx >= 0:
                candidate_sum = s1_vals[idx] + s2
                if candidate_sum > current_best:
                    current_best = candidate_sum
                    current_best_m1 = sums1[idx][1]
                    current_best_m2 = m2
        
        # Update best
        if current_best > best_sum:
            best_sum = current_best
            # Reconstruct indices
            best_indices = []
            # Map mask1 to indices in first half
            for i in range(len(w1)):
                if current_best_m1 & (1 << i):
                    best_indices.append(i)
            # Map mask2 to indices in second half
            for i in range(len(w2)):
                if current_best_m2 & (1 << i):
                    best_indices.append(mid + i)
            
    except Exception:
        # If anything goes wrong, return greedy
        pass

    return best_indices
