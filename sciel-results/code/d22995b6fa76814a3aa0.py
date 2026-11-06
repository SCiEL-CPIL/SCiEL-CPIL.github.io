import time
import math

def construct():
    start_time = time.time()
    # Set a deadline 10 seconds before the 120s limit
    deadline = start_time + 110.0
    
    # Generate powerful numbers up to a limit L
    # L = 5 * 10^9 seems to balance speed and coverage well
    # K ~ 64000, Pairs ~ 2 * 10^9
    # Estimated time ~ 40-60s
    limit = 5000000000 
    
    # Precompute cubes for faster generation
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    pows = set([1])
    max_i = math.isqrt(limit)
    
    # Generate powerful numbers
    # A powerful number is of the form i^2 * j^3
    for i in range(1, max_i + 1):
        sq = i * i
        # Max j such that i^2 * j^3 <= limit
        # j^3 <= limit // sq
        # We can binary search or just loop
        # Since max_j_global is small (~1700), linear scan is fine but we can optimize
        # Actually, limit // sq decreases as i increases.
        
        # Find max j for this i
        # j_max = int((limit // sq) ** (1/3))
        # Using precomputed cubes is faster if we binary search, but simple int power is okay for 70k iterations
        
        # To be safe and fast, just calculate
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
            
        for j in range(1, max_j + 1):
            val = sq * cubes[j-1]
            if val <= limit:
                pows.add(val)
            else:
                break
        
        # Check time during generation (though generation is fast)
        if time.time() > deadline:
            break

    P = sorted(list(pows))
    K = len(P)
    
    # Count sums
    counts = {}
    best_count = 0
    best_sum = 0
    
    P_list = P
    counts_dict = counts
    max_N = 10**18
    
    # Main loop
    for i in range(K):
        ai = P_list[i]
        # Time check
        if time.time() > deadline:
            break
        
        # Inner loop
        # We can potentially skip some j if ai + P_list[j] > max_N
        # Since P_list is sorted, we can break
        for j in range(i, K):
            s = ai + P_list[j]
            if s > max_N:
                break
            
            # Update count
            c = counts_dict.get(s, 0) + 1
            counts_dict[s] = c
            if c > best_count:
                best_count = c
                best_sum = s

    # Retrieve pairs for best_sum
    pairs = []
    pows_set = pows # Use the set for O(1) lookup
    for i in range(K):
        a = P_list[i]
        b = best_sum - a
        if b < a:
            break
        if b in pows_set:
            pairs.append([a, b])
            
    # If no pairs found (should not happen), fallback
    if not pairs:
        # Find any valid pair
        # This case is unlikely
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_sum, 'pairs': pairs}
