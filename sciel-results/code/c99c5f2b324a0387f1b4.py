def construct():
    N = 180
    # Pre-compute all reciprocal triples {a, b, c} with a < b < c
    triples = []
    for a in range(1, N + 1):
        # b must be > a
        # c = ab / (b - a)
        # c > b => ab > b(b-a) => 2a > b
        # c <= N => ab <= N(b-a) => a(b+N) <= Nb => a <= Nb / (b+N)
        max_b = min(N, 2 * a - 1)
        for b in range(a + 1, max_b + 1):
            denom = b - a
            if (a * b) % denom == 0:
                c = (a * b) // denom
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    # Create a mapping from element to list of triple indices containing it
    # Or just keep a set of active triple indices
    active_triples = set(range(len(triples)))
    
    # Count occurrences of each element in active triples
    # We will recompute or update this.
    # Given N=180 and small number of triples, we can just scan.
    
    S = set(range(1, N + 1))
    
    while active_triples:
        # Find the element in the active triples with the highest frequency
        counts = {x: 0 for x in S}
        for idx in active_triples:
            a, b, c = triples[idx]
            if a in S:
                counts[a] += 1
            if b in S:
                counts[b] += 1
            if c in S:
                counts[c] += 1
        
        # Find max count
        max_count = -1
        target = None
        for x, cnt in counts.items():
            if cnt > max_count:
                max_count = cnt
                target = x
        
        if target is None:
            break
            
        # Remove target from S
        S.remove(target)
        
        # Remove all triples containing target from active_triples
        new_active = set()
        for idx in active_triples:
            a, b, c = triples[idx]
            if target not in (a, b, c):
                new_active.add(idx)
        active_triples = new_active

    return list(S)
