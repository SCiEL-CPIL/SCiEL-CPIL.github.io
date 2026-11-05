import time
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two sets
    mid = n // 2
    A = denominators[:mid]
    B = denominators[mid:]
    
    # Generate subset sums as floats for A and B
    # Store as (sum_float, mask)
    sums_A = [(0.0, 0)]
    for i, d in enumerate(A):
        inv = 1.0 / d
        new_sums = []
        for s, m in sums_A:
            new_sums.append((s + inv, m | (1 << i)))
        sums_A.extend(new_sums)
        
    sums_B = [(0.0, 0)]
    for i, d in enumerate(B):
        inv = 1.0 / d
        new_sums = []
        for s, m in sums_B:
            new_sums.append((s + inv, m | (1 << i)))
        sums_B.extend(new_sums)
        
    # Sort by sum
    sums_A.sort(key=lambda x: x[0])
    sums_B.sort(key=lambda x: x[0])
    
    best_sum = -1.0
    best_a_mask = 0
    best_b_mask = 0
    best_frac = None
    
    # Two-pointer search
    # We want max a + b < 1.0
    # Iterate b from smallest to largest?
    # If b increases, max allowed a decreases.
    # So iterate b in ascending order, and a pointer moves from high to low.
    
    i = len(sums_A) - 1
    for j in range(len(sums_B)):
        b_sum, b_mask = sums_B[j]
        # We need a_sum + b_sum < 1.0
        # So a_sum < 1.0 - b_sum
        target = 1.0 - b_sum
        while i >= 0 and sums_A[i][0] > target:
            i -= 1
        if i < 0:
            break
        a_sum, a_mask = sums_A[i]
        total = a_sum + b_sum
        if total > best_sum:
            best_sum = total
            best_a_mask = a_mask
            best_b_mask = b_mask
            
            # Check time
            if time.time() > deadline - 10:
                break
                
    if best_sum < 0:
        return [0] # Fallback
        
    # Reconstruct candidate
    selected = []
    for i in range(mid):
        if best_a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if best_b_mask & (1 << i):
            selected.append(mid + i)
            
    # Verify with Fraction
    # It is possible that float sum < 1.0 but exact sum >= 1.0
    # Or float sum was not the true max due to precision.
    # We should check this candidate and potentially neighbors.
    
    # Calculate exact fraction for the best float candidate
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    
    # If it's invalid (>= 1), we need to find the next best.
    # This is tricky. 
    # However, the probability of float error causing a valid subset to be rejected or an invalid one to be accepted as "best" is low if the gap is large.
    # But we want the *strictly below 1*.
    # If total_frac >= 1, we must discard it.
    # How to find the next best?
    # We can keep a list of top K candidates from the MITM.
    
    # Let's modify the search to keep the top 100 candidates.
    # This is more robust.
    
    # Re-do search to collect top candidates
    # Instead of just tracking the max, we can collect candidates that are close to 1.
    # Or, simply, if the best fails, we can try removing the smallest element from the subset?
    # If sum >= 1, removing an element reduces the sum.
    # The best valid subset is likely close to the best invalid subset.
    
    # Strategy:
    # 1. Get the best float candidate.
    # 2. Verify.
    # 3. If invalid, try removing each element one by one and verify.
    # 4. Pick the best valid one.
    # 5. Also, we might have missed a valid subset that was slightly lower in float sum but valid.
    #    But since we want to maximize sum < 1, the best valid one is very close to 1.
    #    The float search finds the global max of (a+b).
    #    If that max is >= 1 (exact), the next best float sum might be valid.
    #    So we should check the next few best float sums.
    
    # Let's collect the top 20 candidates from the two-pointer scan.
    # The two-pointer scan visits pairs (i, j).
    # The sums are not strictly sorted globally in the scan order, but we can collect candidates with sum > best_sum - epsilon.
    
    # Simpler: Just check the best one. If it fails, check the next best one found in the scan?
    # The scan doesn't naturally produce the "next best" easily because i moves.
    
    # Alternative:
    # If the best candidate is invalid, it means sum >= 1.
    # We can try to "fix" it by removing elements.
    # But maybe there's a completely different subset that is valid and closer to 1.
    # Given the density of subset sums, it's likely that the best valid one is very close to the best invalid one.
    
    # Let's implement a local search around the best candidate.
    # If invalid, remove the element with the largest value? No, remove the smallest value to minimize loss?
    # Actually, if sum > 1, we want to reduce sum to < 1.
    # Removing the smallest term reduces sum the least.
    
    if total_frac >= 1:
        # Try removing each element
        best_valid_frac = None
        best_valid_indices = None
        
        # Also, we should consider that the "next best" MITM candidate might be valid.
        # Let's just try removing elements from the current best.
        for k in selected:
            new_selected = [x for x in selected if x != k]
            if not new_selected: continue
            new_frac = total_frac - Fraction(1, denominators[k])
            if new_frac < 1 and (best_valid_frac is None or new_frac > best_valid_frac):
                best_valid_frac = new_frac
                best_valid_indices = new_selected
                
        if best_valid_indices:
            return best_valid_indices
        else:
            # Fallback to greedy
            selected_greedy = []
            total_greedy = Fraction(0)
            for k in range(n):
                if total_greedy + Fraction(1, denominators[k]) < 1:
                    selected_greedy.append(k)
                    total_greedy += Fraction(1, denominators[k])
            return selected_greedy
    else:
        return selected
