import time
import bisect
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 10000000
    target_num, target_den = 3, 2
    
    # 1. Compute sigma(n) for all n in [1, N] using NumPy
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        # Fallback to a simple known pair if sieve is too slow (unlikely)
        return [11, 6]

    # 2. Group numbers by their sigma value
    # We only care about sigma values that appear more than once
    sigma_groups = {}
    for n in range(1, N + 1):
        s_val = int(sigma[n])
        if s_val in sigma_groups:
            sigma_groups[s_val].append(n)
        else:
            sigma_groups[s_val] = [n]
            
    if time.time() > deadline:
        # Find any pair if grouping took too long
        for s_val, group in sigma_groups.items():
            if len(group) > 1:
                return [group[0], group[1]]
        return [11, 6]

    # 3. Search for pairs (a, b) in each group minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (3b)
    # We want to minimize this value.
    # Since b > 0, minimizing |2a - 3b| / (3b) is roughly equivalent to
    # finding b close to 2a/3.
    
    best_error_num = 1
    best_error_den = 1
    best_pair = (11, 6) # Default valid pair: sigma(11)=12, sigma(6)=12
    
    for s_val, group in sigma_groups.items():
        if len(group) < 2:
            continue
            
        # Sort the group to use bisect
        group.sort()
        
        # For each a in group, find b in group closest to 2a/3
        # We iterate over all a, but we can optimize by checking if current best error
        # is already very small.
        
        for a in group:
            # Target b is approximately 2a/3
            target_b = (2 * a) / 3.0
            
            # Find insertion point in sorted group
            idx = bisect.bisect_left(group, target_b)
            
            # Check neighbors: idx-1, idx, idx+1
            candidates = []
            if idx > 0:
                candidates.append(group[idx - 1])
            if idx < len(group):
                candidates.append(group[idx])
            if idx + 1 < len(group):
                candidates.append(group[idx + 1])
            
            for b in candidates:
                if a == b:
                    continue
                
                # Calculate error: |a/b - 3/2| = |2a - 3b| / (3b)
                # Numerator: |2a - 3b|
                # Denominator: 3b
                err_num = abs(2 * a - 3 * b)
                err_den = 3 * b
                
                # Compare with best_error
                # err_num/err_den < best_error_num/best_error_den
                # => err_num * best_error_den < best_error_num * err_den
                if err_num * best_error_den < best_error_num * err_den:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_pair = (a, b)
                    
                    # If error is 0, we can't score it, but we should avoid it.
                    # The problem states "raw value of exactly 0 cannot be scored".
                    # So we should look for the next best if err_num == 0.
                    # However, if a/b == 3/2 exactly, then 2a = 3b.
                    # This implies a is a multiple of 3 and b is a multiple of 2.
                    # If we find such a pair, we should probably pick another one
                    # or handle it. But the task says "Minimize |a/b-3/2|".
                    # If the minimum is 0, the score is invalid.
                    # So we must ensure the returned pair does NOT have a/b == 3/2.
                    # Let's check if err_num == 0. If so, we should not update best_pair
                    # unless we have no other choice? No, we must return a valid pair.
                    # So if err_num == 0, we skip updating best_pair to 0-error,
                    # but we need to keep the best NON-ZERO error.
                    # Wait, if the global minimum is 0, we must return the next smallest.
                    # So we should only update if err_num > 0.
                    # Let's re-evaluate.
                    # If err_num == 0, this pair is invalid for scoring.
                    # We should not set best_pair to this.
                    # But what if all pairs have error 0? Unlikely.
                    # So, we only update best_pair if err_num > 0.
                    
                    # Correction: The logic above updated best_pair even if err_num was 0.
                    # We need to fix that.
                    pass
                    
        if time.time() > deadline:
            break

    # Refine the comparison logic to exclude err_num == 0
    # The loop above might have set best_pair to a 0-error pair.
    # Let's re-run the comparison check for the final result or ensure
    # we never select a 0-error pair as the 'best' if we can find a non-zero one.
    
    # Actually, it's safer to just check the final best_pair.
    # If best_pair has error 0, we need to find the next best.
    # But it's easier to just not update best_pair if err_num == 0.
    
    # Let's rewrite the inner loop logic to be safe.
    
    best_error_num = 1
    best_error_den = 1
    best_pair = (11, 6)
    
    for s_val, group in sigma_groups.items():
        if len(group) < 2:
            continue
            
        group.sort()
        
        for a in group:
            target_b = (2 * a) / 3.0
            idx = bisect.bisect_left(group, target_b)
            
            candidates = []
            if idx > 0:
                candidates.append(group[idx - 1])
            if idx < len(group):
                candidates.append(group[idx])
            if idx + 1 < len(group):
                candidates.append(group[idx + 1])
            
            for b in candidates:
                if a == b:
                    continue
                
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    continue # Invalid for scoring, skip
                
                err_den = 3 * b
                
                if err_num * best_error_den < best_error_num * err_den:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_pair = (a, b)
                    
        if time.time() > deadline:
            break

    return [best_pair[0], best_pair[1]]
