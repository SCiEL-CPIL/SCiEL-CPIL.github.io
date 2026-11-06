def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    for a in range(1, n + 1):
        # Skip self-inverse residues
        if a == 1 or a == 1008:
            continue
        # Find max b >= a such that [a,b] is valid
        # [a,b] valid iff for all x in [a,b], inv[x] not in [a,b]
        b = a
        valid = True
        for b_candidate in range(a, n + 1):
            # Check if adding b_candidate breaks validity
            # We need: for all x in [a, b_candidate], inv[x] < a or inv[x] > b_candidate
            # This is equivalent to: no x in [a, b_candidate] has inv[x] in [a, b_candidate]
            # We can check incrementally, but for simplicity with small n, check the new element
            x = b_candidate
            if a <= inv[x] <= b_candidate:
                # Adding b_candidate makes it invalid (b_candidate's inverse is in interval)
                # Also need to check if any earlier x in [a, b_candidate-1] has inv in [a, b_candidate]
                # But we already validated [a, b_candidate-1], so only new conflict is:
                # 1. inv[b_candidate] in [a, b_candidate] (checked above)
                # 2. Some x in [a, b_candidate-1] with inv[x] == b_candidate
                #    i.e., x = inv[b_candidate], which we already checked via condition 1
                break
            # Also check: is there any x in [a, b_candidate-1] whose inverse is b_candidate?
            # That's x = inv[b_candidate]. If inv[b_candidate] in [a, b_candidate-1], then conflict.
            # But that's the same as inv[b_candidate] in [a, b_candidate] which we checked.
            b = b_candidate
        
        length = b - a + 1
        if length > best_len:
            best_len = length
            best_start = a
    
    return [best_start, best_len]
