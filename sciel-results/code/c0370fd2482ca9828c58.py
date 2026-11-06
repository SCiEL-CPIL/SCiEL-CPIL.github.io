"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 40

# EVOLVE-BLOCK-START

def _validate(edges, cycles):
    """Validate that all cycle edges exist in the edge set and vertices are distinct."""
    edge_set = set()
    for e in edges:
        u, v = e
        edge_set.add((min(u, v), max(u, v)))
    
    for k, cyc in enumerate(cycles):
        length = k + 3  # cycles[0] is length 3, cycles[1] is length 4, etc.
        if len(cyc) != length:
            return False
        if len(set(cyc)) != length:
            return False
        for i in range(length):
            u = cyc[i]
            v = cyc[(i + 1) % length]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True


def _construct_twin():
    """
    Twin construction: Two cliques K20 on {0..19} and {20..39}, 
    connected by perfect matching (i, i+20) for i in 0..19.
    
    For cycle of length k:
    - If k <= 20: take k consecutive vertices from clique A (cyclic order 0,1,...,19)
    - If k > 20: take (k-20) vertices from clique A and (20 - (k-20)) = (40-k) vertices from clique B,
      but we need to ensure connectivity. Actually, let's think more carefully.
    
    Actually, a simpler approach:
    - The graph has edges: all pairs within 0..19, all pairs within 20..39, and matching (i, i+20).
    
    For a cycle of length k:
    - If k <= 20: use k vertices from clique A: 0, 1, 2, ..., k-1. Since they're all in clique A, 
      the cycle 0->1->...->(k-1)->0 works.
    - If k = 21: use all 20 from A and 1 from B. E.g., 0,1,...,19,20. 
      Edges: (0,1),...,(18,19) in A, (19,20) is matching? No, matching is (i, i+20). 
      So (19, 39) is matching, not (19,20). Hmm.
    
    Let me reconsider. The matching connects i to i+20. So vertex 19 connects to 39, vertex 0 to 20, etc.
    
    For k=21: vertices 0..19 (clique A) and vertex 20. Edges needed: cycle through all 21.
    Path: 0,1,2,...,19,20. Edges: (0,1),...,(18,19) all in A. (19,20)? Not an edge. 
    (19,39) is matching. (0,20) is matching.
    
    So for k=21: 0,1,2,...,19,20 -> need (19,20) which doesn't exist.
    
    Alternative: 0,1,...,18,19,20. Still need (19,20).
    
    Let's try: 0,20,1,2,...,19. Edges: (0,20) matching, (20,1)? No, 20 is in B, 1 in A. Not an edge.
    
    Hmm, this twin construction with perfect matching is tricky for intermediate cycle lengths.
    
    Let me try a different approach: use a Hamiltonian cycle as backbone plus chords.
    
    Actually, let me just use the known good approach: 40-cycle + 19 chords from a hub.
    """
    # Use a 40-cycle: 0-1-2-...-39-0
    # Plus 19 chords from vertex 0 to vertices 2..20 (or similar)
    
    # The key insight from the knowledge: a chord from a hub on a Hamiltonian cycle 
    # can certify two cycle lengths.
    
    # Let's use vertex 0 as hub. Add chords (0, i) for i in some set.
    # The 40-cycle gives us cycle of length 40: 0,1,2,...,39
    # A chord (0, k) creates a cycle of length k+1: 0,1,2,...,k (using cycle edges 0-1-...-k and chord k-0)
    # And also a cycle of length 40-k: 0,k,k+1,...,39 (using chord 0-k and cycle edges k-(k+1)-...-39-0)
    
    # So chord (0, k) gives cycles of length k+1 and 40-k.
    # We need all lengths 3..40.
    # Length 40: the Hamiltonian cycle itself.
    # Length k+1 for k=1..39: but k=1 gives length 2 (invalid), k=2 gives length 3, etc.
    # Length 40-k for k=1..39: k=1 gives 39, k=2 gives 38, etc.
    
    # We need to cover lengths 3..39 (length 40 is the Hamiltonian cycle).
    # Chord (0, k) covers lengths k+1 and 40-k.
    # For k=2: covers 3 and 38
    # For k=3: covers 4 and 37
    # For k=4: covers 5 and 36
    # ...
    # For k=19: covers 20 and 21
    # For k=20: covers 21 and 20 (same as k=19 essentially)
    
    # So chords (0, 2) through (0, 19) cover:
    # Lengths: 3,4,...,20 (from k+1) and 38,37,...,21 (from 40-k)
    # That's lengths 3..20 and 21..38. Missing: 39.
    
    # Chord (0, 1) would give length 2 and 39. Length 2 is invalid, but 39 is useful.
    # But (0,1) is already a cycle edge, so it's not an extra chord.
    
    # So we need to cover length 39 separately.
    # Length 39: remove one vertex from the Hamiltonian cycle. E.g., remove vertex 20:
    # Cycle: 0,1,2,...,19,21,22,...,39. Edges: all in the 40-cycle except (19,20) and (20,21), 
    # but we need (19,21) which is not an edge.
    
    # Alternatively, use a chord to help. If we have chord (0, 39), that's the same as the cycle edge (39,0).
    
    # Let me add one more chord to cover 39. E.g., chord (1, 21) or something.
    
    # Actually, let's just add chords (0, k) for k in 2..20. That's 19 chords.
    # Total edges: 40 (cycle) + 19 (chords) = 59.
    # This covers lengths 3..20 and 21..38. Missing 39.
    
    # For length 39: we can use vertices 0..18 and 20..39 (skip 19).
    # Need a cycle through these 39 vertices. 
    # Path: 0,1,...,18, then jump to 20? (18,20) not an edge.
    # With chord (0, 19): 0,19,20,21,...,39,1,2,...,18. 
    # Edges: (0,19) chord, (19,20) cycle, (20,21)...(38,39) cycle, (39,0) cycle, (0,1) cycle, (1,2)...(17,18) cycle, (18,0)? Not an edge.
    
    # Hmm. Let me try: 0,1,...,18,19 is not allowed since we skip 19.
    
    # Let's skip vertex 39 instead. Cycle: 0,1,2,...,38.
    # Edges: (0,1),...,(37,38) all cycle edges. (38,0)? Not an edge unless we have chord (0,38).
    # We don't have chord (0,38) in our set (we have 0,2 through 0,20).
    
    # OK, let me add chord (0, 38) instead of one of the others, or add an extra chord.
    
    # Actually, let's just use 20 chords: (0, k) for k in 2..21. That's 20 chords, 60 edges total.
    # k=2..21: covers k+1 = 3..22 and 40-k = 38..19.
    # So lengths 3..22 and 19..38. Union: 3..38. Still missing 39.
    
    # I think the cleanest is: 40-cycle + chords (0,k) for k in 2..20 (19 chords) = 59 edges,
    # and handle length 39 with a specific cycle that uses existing edges.
    
    # For length 39, skip vertex 1: cycle 0,2,3,...,39.
    # Edges: (0,2) chord, (2,3)...(38,39) cycle, (39,0) cycle. All exist!
    
    # Great! So:
    # - Length 40: 0,1,2,...,39
    # - Length 39: 0,2,3,...,39 (skip 1)
    # - Length k for 3<=k<=38: use chord (0, k-1) to get cycle 0,1,...,k-1 (length k)
    #   Wait, chord (0, k-1) gives cycle 0,1,...,k-1 of length k. But we need k-1 >= 2, so k >= 3. Good.
    #   And for k=38: chord (0,37)? We only have chords up to (0,20). 
    #   Chord (0, k) gives cycle of length k+1: 0,1,...,k. So for length 38, need chord (0,37). Not in our set.
    #   But chord (0, k) also gives cycle of length 40-k: 0,k,k+1,...,39. 
    #   For k=2: length 38: 0,2,3,...,39. That's 38 vertices. 
    #   For k=3: length 37: 0,3,4,...,39.
    #   ...
    #   For k=19: length 21: 0,19,20,...,39.
    
    # So with chords (0,2)..(0,20):
    # Length k+1 for k=2..20: lengths 3..21 via cycle 0,1,...,k
    # Length 40-k for k=2..20: lengths 38..20 via cycle 0,k,k+1,...,39
    # Union of lengths: 3..21 and 20..38 = 3..38
    # Plus length 39: 0,2,3,...,39
    # Plus length 40: 0,1,...,39
    
    # All covered!
    
    edges = []
    # 40-cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    # Chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.append((0, k))
    
    # Build cycles
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Length 40: full cycle
            cycles.append(list(range(N)))
        elif length == N - 1:
            # Length 39: skip vertex 1
            cycles.append([0] + list(range(2, N)))
        elif length <= 21:
            # Length k: use chord (0, k-1), cycle 0,1,...,k-1
            k = length
            cycles.append(list(range(k)))
        else:
            # Length k > 21: use chord (0, 40-k), cycle 0, (40-k), (40-k+1), ..., 39
            k = length
            j = N - k  # this is the chord endpoint
            # j is in range 2..18 for k in 22..38
            cycle = [0] + list(range(j, N))
            cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    result = _construct_twin()
    
    # Validate
    if _validate(result['edges'], result['cycles']):
        return result
    
    # Fallback to seed if something goes wrong
    return {'edges': [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)],
            'cycles': [list(range(k)) for k in range(3, N + 1)]}

# EVOLVE-BLOCK-END
