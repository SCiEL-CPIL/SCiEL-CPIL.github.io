    # Greedy initialization for Type 2s
    P = all_mask
    greedy_set = []
    while P:
        # Pick vertex with max degree in P
        # Approximate: pick first?
        # Let's pick one with high degree.
        # To avoid O(N^2) scan every time, we can just pick one.
        # Or sort once?
        
        # Simple greedy:
        # Find i in P with max (P & masks[i]).bit_count()
        # This is O(|P|^2) per step. Total O(N^3). 5280^3 is too big.
        
        # Better: Random restarts or just pick first available.
        # Let's just pick the first vertex in P.
        lsb = P & -P
        i = lsb.bit_length() - 1
        greedy_set.append(i)
        P = P & masks[i]
        
    best_size[0] = len(greedy_set)
    best_set[:] = greedy_set[:]
