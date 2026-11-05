<<<<<<< SEARCH
    # 3. Local Search with removal and perturbation
    # Use a more efficient scan that doesn't rebuild the full index list
    while time.time() < deadline:
=======
    # 3. Aggressive initial removal pass with redundancy-based ordering
    pos_unique_count = [0] * L
    for i in range(L):
        s = max(0, i - n + 1)
        e = min(i, L - n)
        for k in range(s, e + 1):
            p = tuple(S[k:k+n])
            if counts.get(p, 0) == 1:
                pos_unique_count[i] += 1
    
    removable_order = sorted(range(L), key=lambda x: pos_unique_count[x])
    
    for pos in removable_order:
        if time.time() > deadline:
            break
        if pos < L and try_remove(pos):
            pass
    
    # 4. Local Search with removal and perturbation
    while time.time() < deadline:
>>>>>>> REPLACE
