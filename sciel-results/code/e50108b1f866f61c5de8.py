import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the 40-cycle rim (40 edges)
    edges_set = set()
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))

    # Chords to use: d in {2..10, 21..29}
    # Total 18 chords.
    # Primary lengths covered:
    # d=2..10 -> 3..11 and 31..39
    # d=21..29 -> 22..30 and 12..20
    # Missing: 21. Covered by 2-chord cycle (0,2)-(0,21).
    chord_ds = list(range(2, 11)) + list(range(21, 30))
    
    chords = []
    for d in chord_ds:
        e = norm_edge(0, d)
        if e not in edges_set:
            edges_set.add(e)
            chords.append((0, d, d))

    # Build cycles for lengths 3..40
    all_cycles = {N: list(range(N))}  # length 40: full rim

    # Map length to chord d
    # If k = d+1, d = k-1
    # If k = 41-d, d = 41-k
    for k in range(3, N):
        if k == 21:
            # Special case: 2-chord cycle using d=2 and d=21
            # Cycle: 0-2-3-...-21-0
            all_cycles[k] = [0] + list(range(2, 22))
        else:
            d1 = k - 1
            d2 = 41 - k
            # One of d1 or d2 must be in chord_ds
            d = None
            if d1 in chord_ds:
                d = d1
            elif d2 in chord_ds:
                d = d2
            else:
                # Should not happen
                continue
            
            if d == d1:
                # Forward cycle: 0-1-...-d
                all_cycles[k] = list(range(0, d + 1))
            else:
                # Backward cycle: 0-d-(d+1)-...-39
                all_cycles[k] = [0] + list(range(d, N))

    # Verify all 38 cycles (lengths 3..40) are present and valid
    valid = True
    for k in range(3, N + 1):
        cyc = all_cycles.get(k)
        if cyc is None or len(cyc) != k or len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            if norm_edge(cyc[i], cyc[(i + 1) % k]) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if valid:
        return {'edges': [list(e) for e in edges_set],
                'cycles': [all_cycles[k] for k in range(3, N + 1)]}
    else:
        # Fallback: 59-edge solution
        edges_set_fallback = set(edges_set)
        # Add missing chords to make it 59 edges?
        # Actually, if valid is false, we should return a known good solution.
        # The 59-edge solution is safe.
        ef = set()
        for i in range(N):
            ef.add(norm_edge(i, (i + 1) % N))
        for d in range(2, 21):
            ef.add(norm_edge(0, d))
        
        af = {N: list(range(N))}
        for d in range(2, 21):
            len_fwd = d + 1
            len_bwd = N - d + 1
            if len_fwd not in af:
                af[len_fwd] = list(range(0, d + 1))
            if len_bwd not in af:
                af[len_bwd] = [0] + list(range(d, N))
                
        return {'edges': [list(e) for e in ef],
                'cycles': [af[k] for k in range(3, N + 1)]}
