import time
from math import lcm
import bisect
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    # Split into two halves
    mid = n // 2
    half1_idx = list(range(mid))
    half2_idx = list(range(mid, n))
    
    w1 = [weights[i] for i in half1_idx]
    w2 = [weights[i] for i in half2_idx]

    def gen_sums(weights_list):
        """Generate all subset sums < L"""
        sums = [0]
        for w in weights_list:
            if check_time():
                break
            # Optimization: only add if sum < L
            new_sums = []
            for s in sums:
                ns = s + w
                if ns < L:
                    new_sums.append(ns)
            sums.extend(new_sums)
        return sums

    def find_mask_for_sum(target, weights_list, indices_offset):
        """Find a subset of weights_list that sums to target. Returns list of global indices."""
        if target == 0:
            return []
        
        # Use a dictionary to store sum -> mask? 
        # Or just generate all and check.
        # Since N is small (~24), generating all is fine.
        # We need to return indices.
        
        # Efficient way: 
        # We can use a set of achievable sums and backtrack?
        # Or just generate all masks.
        
        n_items = len(weights_list)
        # If target is not achievable, return None
        # We know it is achievable because we found it in the sums list.
        
        # Generate sums and masks
        # To save memory, we don't need to store all if we just want one.
        # But we need to find the one matching target.
        
        # We can do a DFS?
        # Or iterative generation with early exit?
        # Iterative generation creates a list of (sum, mask).
        
        pairs = [(0, 0)]
        for i, w in enumerate(weights_list):
            if check_time():
                break
            bit = 1 << i
            current_len = len(pairs)
            for j in range(current_len):
                s, m = pairs[j]
                ns = s + w
                if ns == target:
                    # Found it!
                    # Convert mask to indices
                    indices = []
                    for k in range(n_items):
                        if m | bit & (1 << k): # Wait, mask logic
                            pass
                    # Correct mask conversion
                    full_mask = m | bit
                    indices = []
                    for k in range(n_items):
                        if full_mask & (1 << k):
                            indices.append(indices_offset + k)
                    return indices
                elif ns < L:
                    pairs.append((ns, m | bit))
                # If ns > L, we skip it, but we must be careful: 
                # If we skip it, we might miss a combination that sums to target later?
                # No, if ns > L, and all weights are positive, any superset will be > L.
                # And target < L. So we can safely prune.
        
        return None # Should not happen

    # 1. Generate sums
    print("Generating sums...")
    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)
    
    if check_time():
        return [0] # Fallback

    print(f"Sum counts: {len(sums1)}, {len(sums2)}")

    # 2. Sort
    sums1.sort()
    sums2.sort()
    
    # 3. Find Max Sum < L
    # Two pointer technique
    # s1 from smallest to largest
    # s2 from largest to smallest
    j = len(sums2) - 1
    best_sum = -1
    
    for s1 in sums1:
        # We need s1 + s2 < L
        # So s2 < L - s1
        # Since s1 is increasing, L - s1 is decreasing.
        # So j (index of max valid s2) should decrease or stay same.
        
        while j >= 0 and s1 + sums2[j] >= L:
            j -= 1
            
        if j < 0:
            # No valid s2 for this s1. Since s1 is increasing, 
            # L - s1 will get even smaller, so no future s1 will work either.
            break
            
        current_sum = s1 + sums2[j]
        if current_sum > best_sum:
            best_sum = current_sum

    if best_sum == -1:
        # Fallback: single item
        max_w = max(weights)
        best_sum = max_w
        # Find index of max_w
        best_indices = [weights.index(max_w)]
        return best_indices

    # 4. Find the specific pair (s1, s2)
    # We know best_sum.
    # We need s1 in sums1, s2 in sums2 such that s1 + s2 = best_sum.
    
    # Convert sums2 to a set for O(1) lookup
    set2 = set(sums2)
    
    target_s1 = None
    target_s2 = None
    
    for s1 in sums1:
        s2_needed = best_sum - s1
        if s2_needed in set2:
            target_s1 = s1
            target_s2 = s2_needed
            break
            
    if target_s1 is None:
        # Should not happen, but fallback
        return [0]

    # 5. Find indices for target_s1 and target_s2
    print("Reconstructing indices...")
    idx1 = find_mask_for_sum(target_s1, w1, 0)
    idx2 = find_mask_for_sum(target_s2, w2, mid)
    
    if idx1 is None or idx2 is None:
        # Fallback
        return [0]
        
    final_indices = idx1 + idx2
    
    # Final Validation
    if not final_indices:
        return [0]
        
    s = sum(weights[i] for i in final_indices)
    if s >= L:
        # This shouldn't happen, but if it does, try to fix
        # Remove one item
        final_indices.pop()
        s = sum(weights[i] for i in final_indices)
        
    return final_indices
