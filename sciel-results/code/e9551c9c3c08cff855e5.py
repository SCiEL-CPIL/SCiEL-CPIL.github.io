import time
import sys
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate powerful numbers
    # Powerful numbers are of the form u^2 * v^3
    # We need to cover sums up to N_max.
    # Let's aim for N_max = 10^18.
    # If we generate powerful numbers up to P_max, we can find sums up to 2*P_max.
    # However, we want to find N with many representations.
    # The density of powerful numbers is roughly N^{1/3} / 2.
    # To get many representations, we need a large pool of powerful numbers.
    
    # Let's generate powerful numbers up to 10^12.
    # This allows sums up to 2*10^12, which is well within 10^18.
    # But wait, if we only sum numbers up to 10^12, the maximum sum is 2*10^12.
    # Is it better to generate larger powerful numbers?
    # The number of powerful numbers up to X is ~ c * X^{1/3}.
    # If X = 10^12, K ~ 10^4. Pairs ~ 10^8. This is feasible.
    # If X = 10^18, K ~ 10^6. Pairs ~ 10^12. Too slow.
    
    # Let's try a larger bound for powerful numbers to increase the chance of collisions (sums).
    # Actually, the "collision" (same sum) is what we want.
    # The probability of two random powerful numbers summing to the same value increases with the density.
    # But powerful numbers are sparse.
    
    # Let's stick to generating powerful numbers up to 10^12 or 10^13.
    # 10^13: K ~ 20,000. Pairs ~ 2*10^8. Might be too slow in Python (200M iterations).
    # 10^12: K ~ 10,000. Pairs ~ 5*10^7. ~10-20 seconds in Python.
    
    P_MAX = 10**12
    
    powerful_set = set()
    max_v = isqrt(isqrt(P_MAX)) + 2 # v^3 <= P_MAX => v <= P_MAX^(1/3)
    # Actually v^3 <= P_MAX => v <= P_MAX^(1/3).
    # u^2 <= P_MAX / v^3 => u <= sqrt(P_MAX / v^3).
    
    # Iterate v
    v = 1
    while v**3 <= P_MAX:
        v_cubed = v**3
        max_u = isqrt(P_MAX // v_cubed)
        u = 1
        while u <= max_u:
            p = u*u * v_cubed
            if p <= P_MAX:
                powerful_set.add(p)
            u += 1
            # Time check
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
        v += 1
    
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    # 2. Hash-map frequency counting
    # We want to count sums a+b where a, b in powerful_list.
    # To save time, we can restrict a <= b to avoid double counting and division by 2.
    # But we need to find the max count.
    
    # Optimization: Only iterate a up to a certain limit?
    # If we iterate all pairs, it's K^2/2.
    # K ~ 10,000 => 50,000,000 iterations.
    # In Python, 50M iterations takes ~5-10 seconds.
    # We have ~145 seconds. We can afford K up to ~30,000-40,000.
    
    # Let's increase P_MAX if we have time?
    # For now, let's run with P_MAX = 10^12.
    
    counts = {}
    
    # Nested loop
    # To speed up, we can use local variables
    p_list = powerful_list
    K_local = K
    
    for i in range(K_local):
        a = p_list[i]
        # If a is too large, the sums will be large.
        # We don't have a strict upper bound on N other than 10^18.
        # But if we want many representations, N should be in the "middle" of the range of sums.
        # Max sum is 2 * P_MAX = 2 * 10^12.
        
        # We can break early if a is very large?
        # No, we need all pairs.
        
        for j in range(i, K_local):
            b = p_list[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            
            # Time check every 10000 iterations?
            # Better to check less frequently to avoid overhead.
            if (j & 0x3FF) == 0 and time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break
    
    # 3. Find the best N
    best_N = 0
    best_count = 0
    
    # We need to reconstruct pairs for the best N.
    # First, find the N with the highest count.
    # Then, verify and collect pairs.
    
    # It's possible that the hash map has floating point issues? No, integers.
    
    # Get top candidates
    # Sorting a dict of ~10^7 entries might be slow.
    # We can just iterate and keep track of max.
    
    max_val = 0
    max_key = 0
    for s, c in counts.items():
        if c > max_val:
            max_val = c
            max_key = s
    
    if max_val == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max_key
    
    # 4. Verify and collect pairs for best_N
    # We need to find all pairs (a, b) such that a+b = best_N, a<=b, a,b in powerful_set.
    # We can iterate through powerful_list again.
    
    pairs = []
    p_set = powerful_set
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    # The count from hash map might be slightly off if we broke early in the loop.
    # But the "max_val" is a lower bound on the true count for that N?
    # No, if we broke early, we missed some pairs.
    # So the true count for best_N is at least max_val.
    # But another N might have a higher true count if we missed its pairs.
    # However, we broke early only if we were close to the deadline.
    # If we finished the loop, the counts are exact.
    
    # Let's check if we finished the loop.
    # We don't have a flag. Let's assume if we are here, we have a good candidate.
    # To be safe, we can do a second pass for the top 10 candidates?
    # But finding top 10 requires sorting.
    
    # Given the time constraint, returning the best found is a reasonable heuristic.
    
    return {'N': best_N, 'pairs': pairs}
