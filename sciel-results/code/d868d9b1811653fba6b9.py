import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the 40-cycle rim
    edges_set = set()
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))

    # Add 19 chords with even distances 2, 4, ..., 38 from vertex 0
    # Chord (0, d) for even d in 2..38 (step 2)
    # This gives distances: 2,4,6,...,38
    # Cycle lengths covered by chord (0,d): d+1 and N-d+1
    # For d=2: cycles 3 and 39
    # For d=4: cycles 5 and 37
    # For d=6: cycles 7 and 35
    # ...
    # For d=38: cycles 39 and 3 (same as d=2, but we only need each length once)
    # Actually let's be more careful:
    # d=2 -> 3, 39
    # d=4 -> 5, 37
    # d=6 -> 7, 35
    # d=8 -> 9, 33
    # d=10 -> 11, 31
    # d=12 -> 13, 29
    # d=14 -> 15, 27
    # d=16 -> 17, 25
    # d=18 -> 19, 23
    # d=20 -> 21, 21 (both paths give 21)
    # d=22 -> 23, 19 (same as d=18)
    # ...
    # We need lengths 3,4,5,...,39
    # The even distances give odd cycle lengths (d+1 is odd when d is even)
    # We're missing even cycle lengths: 4,6,8,...,38
    # 
    # Let me reconsider. I need to cover ALL lengths 3..39.
    # 
    # Better approach: use a mix of chords.
    # Actually, the known result is that a 40-cycle plus 19 carefully chosen chords
    # can be pancyclic. Let me use a different strategy:
    #
    # Use chords that create specific missing lengths.
    # 
    # Alternative: Use the "wheel-like" construction but optimized.
    # Actually, let me just use a proven construction:
    # 
    # The simplest reliable approach: use vertex 0 connected to vertices 2,3,4,...,39
    # That's the fan with 39 edges from 0 plus 39 rim edges = 78 edges. Too many.
    #
    # Let me think differently. We need 37 cycle lengths (3-39).
    # Each chord can create at most 2 new cycle lengths.
    # So minimum chords = ceil(37/2) = 19.
    #
    # Let me construct chords that cover pairs of lengths:
    # Length 3 needs dist 2 chord
    # Length 4 needs dist 3 chord  
    # Length 5 needs dist 4 chord
    # ...
    # Length 39 needs dist 38 chord (or dist 2, same thing)
    #
    # Pair them up:
    # (3, 38): need dist 2 and dist 37 chords
    # (4, 37): need dist 3 and dist 36 chords
    # (5, 36): need dist 4 and dist 35 chords
    # (6, 35): need dist 5 and dist 34 chords
    # (7, 34): need dist 6 and dist 33 chords
    # (8, 33): need dist 7 and dist 32 chords
    # (9, 32): need dist 8 and dist 31 chords
    # (10, 31): need dist 9 and dist 30 chords
    # (11, 30): need dist 10 and dist 29 chords
    # (12, 29): need dist 11 and dist 28 chords
    # (13, 28): need dist 12 and dist 27 chords
    # (14, 27): need dist 13 and dist 26 chords
    # (15, 26): need dist 14 and dist 25 chords
    # (16, 25): need dist 15 and dist 24 chords
    # (17, 24): need dist 16 and dist 23 chords
    # (18, 23): need dist 17 and dist 22 chords
    # (19, 22): need dist 18 and dist 21 chords
    # (20, 21): need dist 19 and dist 20 chords
    #
    # That's 19 pairs, each needing 2 chords = 38 chords. Too many!
    #
    # Wait, I made an error. A single chord of distance d creates cycles of length d+1 AND N-d+1.
    # So one chord covers TWO lengths simultaneously.
    #
    # Chord of dist 2: covers 3 and 39
    # Chord of dist 3: covers 4 and 38
    # Chord of dist 4: covers 5 and 37
    # Chord of dist 5: covers 6 and 36
    # Chord of dist 6: covers 7 and 35
    # Chord of dist 7: covers 8 and 34
    # Chord of dist 8: covers 9 and 33
    # Chord of dist 9: covers 10 and 32
    # Chord of dist 10: covers 11 and 31
    # Chord of dist 11: covers 12 and 30
    # Chord of dist 12: covers 13 and 29
    # Chord of dist 13: covers 14 and 28
    # Chord of dist 14: covers 15 and 27
    # Chord of dist 15: covers 16 and 26
    # Chord of dist 16: covers 17 and 25
    # Chord of dist 17: covers 18 and 24
    # Chord of dist 18: covers 19 and 23
    # Chord of dist 19: covers 20 and 22
    # Chord of dist 20: covers 21 and 21
    #
    # That's 19 chords covering all lengths 3-39!
    # Chords at distances 2 through 20 from some vertex.
    #
    # Let's use vertex 0 as the hub for these chords.
    # Chord (0, d) for d in 2, 3, 4, ..., 20
    #
    # Total edges: 40 (rim) + 19 (chords) = 59 edges
    # Objective: 59 - 40 = 19

    chords = []
    for d in range(2, 21):  # distances 2 through 20
        u = 0
        v = d % N
        if u == v:
            continue
        e = norm_edge(u, v)
        if e not in edges_set:
            edges_set.add(e)
            chords.append((u, v, d))

    # Now extract explicit cycles for each length 3-40
    cycles = []
    
    # For length 40: the full rim cycle
    cycles.append(list(range(N)))  # 0,1,2,...,39
    
    # For lengths 3-39, we need to find cycles in the graph
    # We'll use the chord structure:
    # Chord (0, d) with distance d on the rim creates:
    #   Cycle A: 0, 1, 2, ..., d, 0 (length d+1) - going forward along rim
    #   Cycle B: 0, d, d+1, ..., 39, 0 (length N-d+1) - going the other way
    
    # Let's map each required length to a chord
    length_to_chord = {}
    for u, v, d in chords:
        # Forward path: 0 -> 1 -> ... -> d (length d edges, d+1 vertices)
        len_a = d + 1
        # Backward path: 0 -> d -> d+1 -> ... -> 39 -> 0 (length N-d edges, N-d+1 vertices)  
        len_b = N - d + 1
        
        if len_a <= N and len_a >= 3:
            length_to_chord[len_a] = (u, v, d, 'forward')
        if len_b <= N and len_b >= 3 and len_b != len_a:
            length_to_chord[len_b] = (u, v, d, 'backward')
        elif len_b <= N and len_b >= 3:
            # If both give same length, still record it
            if len_b not in length_to_chord:
                length_to_chord[len_b] = (u, v, d, 'backward')
    
    # Now build each cycle
    all_cycles = {}
    all_cycles[N] = list(range(N))  # length 40
    
    for k in range(3, N):
        if k in length_to_chord:
            u, v, d, direction = length_to_chord[k]
            if direction == 'forward':
                # Cycle: 0, 1, 2, ..., d, 0
                # Vertices: 0, 1, 2, ..., d
                cyc = list(range(0, d + 1))
            else:
                # Cycle: 0, d, d+1, ..., 39, 0
                # Vertices: 0, d, d+1, ..., 39
                cyc = [0] + list(range(d, N))
            all_cycles[k] = cyc
        else:
            # Fallback: shouldn't happen, but use a simple search
            all_cycles[k] = None
    
    # Verify all cycles are found and valid
    valid = True
    for k in range(3, N + 1):
        if all_cycles.get(k) is None:
            valid = False
            break
        cyc = all_cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if norm_edge(a, b) not in edges_set:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        cycles_out = [all_cycles[k] for k in range(3, N + 1)]
        edges_out = [list(e) for e in edges_set]
        return {'edges': edges_out, 'cycles': cycles_out}
    else:
        # Fallback to fan graph (77 edges, objective 37)
        edges_set = set()
        for i in range(1, N):
            edges_set.add(norm_edge(0, i))
        for i in range(1, N - 1):
            edges_set.add(norm_edge(i, i + 1))
        cycles_fallback = []
        for k in range(3, N + 1):
            cycles_fallback.append(list(range(k)))
        return {'edges': [list(e) for e in edges_set], 'cycles': cycles_fallback}
