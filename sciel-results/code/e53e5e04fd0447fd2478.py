while time.time() < deadline:
    # Try to remove
    removed = False
    for i in range(len(S)):
        c = S[i]
        is_redundant = True
        for w in balls[c]:
            if coverage[w] == 1:
                is_redundant = False
                break
        if is_redundant:
            # Remove c
            for w in balls[c]:
                coverage[w] -= 1
            S.pop(i)
            in_S[c] = False
            removed = True
            break
    if removed:
        continue

    # Try to swap
    # Pick random c_in
    if len(S) == 0: break
    idx_in = np.random.randint(len(S))
    c_in = S[idx_in]
    
    # Find U_in
    U_in = []
    for w in balls[c_in]:
        if coverage[w] == 1:
            U_in.append(w)
            
    if not U_in:
        # Should have been removed
        continue

    # Pick random c_out not in S
    # We can pick random indices from 0..N-1 until we find one not in S
    # Or maintain a list of available codewords.
    # Given N=6561 and |S|~500, picking random is fine (prob of hit is high).
    while True:
        c_out = np.random.randint(N)
        if not in_S[c_out]:
            break
            
    # Check if U_in subset of balls[c_out]
    valid_swap = True
    for w in U_in:
        if w not in balls[c_out]: # This is O(20)
            valid_swap = False
            break
            
    if valid_swap:
        # Perform swap
        # Remove c_in
        for w in balls[c_in]:
            coverage[w] -= 1
        # Add c_out
        for w in balls[c_out]:
            coverage[w] += 1
        # Update S and in_S
        S[idx_in] = c_out
        in_S[c_in] = False
        in_S[c_out] = True
        # Continue to try removals
        continue
