"""Pancyclic graph on 40 vertices - balanced chord construction."""
import time
import random

N = 40
START_TIME = time.time()
DEADLINE = 85  # seconds, keep margin

def elapsed():
    return time.time() - START_TIME

def construct():
    if elapsed() > DEADLINE - 5:
        # Fallback: fan + path
        edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        cycles = [list(range(k)) for k in range(3, N+1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Strategy: Use a base cycle and add chords from multiple centers.
    # Base cycle: 0-1-2-...-39-0
    # For cycle length L, we need a chord that, combined with a path on the base cycle,
    # forms a cycle of exactly L vertices.
    
    # Key insight: If we have chord (a, b) where the distance along the base cycle 
    # (going one way) is d edges, then we get a cycle of length d+1 (the short arc + chord)
    # and a cycle of length N-d+1 (the long arc + chord).
    
    # We need cycles of lengths 3, 4, ..., 39 (37 cycles) plus length 40 (base cycle).
    # Each chord can potentially serve 2 cycle lengths (d+1 and N-d+1 where d is the distance).
    
    # Optimal: pair up lengths. For length L, we need a chord at distance L-1 or N-L+1.
    # So length L pairs with length N+1-L (since (L-1) + (N-L+1) = N).
    # L and N+1-L: 3&38, 4&37, 5&36, ..., 20&21, 21&20, ...
    # So we need chords at distances: 2,3,4,...,19,20 (covering pairs 3-38, 4-37, ..., 20-21)
    # That's 19 chords minimum if each covers exactly one pair.
    
    # But we can be smarter: use chords from different centers so cycles don't share edges.
    
    # Let's try: base cycle + 19 chords, but place them strategically.
    # For distance d, chord (i, i+d mod N) creates cycles of length d+1 and N-d+1.
    
    # To minimize shared edges, spread chords across different vertices.
    
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # We need to cover cycle lengths 3..39.
    # Length L requires a chord at distance L-1 or 41-L (mod 40, take min).
    # Let's assign each length to a specific chord.
    
    # Pair lengths: (3,38), (4,37), (5,36), ..., (20,21)
    # For pair (L, 41-L), we need a chord at distance min(L-1, 40-L+1) = min(L-1, 41-L)
    # L=3: dist 2, L=38: dist 2 -> same chord covers both!
    # So one chord at distance d covers lengths d+1 and N-d+1 = 41-d.
    
    # Distances needed: for L in 3..20, dist = L-1 (covers L and 41-L)
    # Distances: 2, 3, 4, ..., 19, 20
    # That's 19 distinct distances.
    
    # Now, to minimize edge sharing, place chords at different positions.
    # Chord at distance d from center c: (c, c+d) mod N
    # The cycle of length d+1 uses vertices c, c+1, ..., c+d (mod N) and edge (c, c+d)
    # The cycle of length 41-d uses the other arc.
    
    # To minimize overlap, spread centers: use centers 0, 1, 2, ... for different chords.
    
    chords = []
    # For distance d (2..20), place chord from center (d-2) mod N
    # This spreads them out.
    for d in range(2, 21):
        center = (d - 2) % N
        a = center
        b = (center + d) % N
        chords.append((a, b, d))
    
    edge_set = set(base_edges)
    for a, b, d in chords:
        edge_set.add((min(a, b), max(a, b)))
    
    edges = list(edge_set)
    
    # Now construct cycles for each length 3..40
    cycles = []
    
    # For each length L in 3..39:
    # If L <= 21: use chord at distance L-1, cycle is the short arc
    # If L >= 22: use chord at distance 41-L, cycle is the long arc (which has 41-(41-L) = L vertices)
    
    # For L in 3..21:
    #   d = L-1, center = (d-2) % N = (L-3) % N
    #   Cycle: center, center+1, ..., center+d (mod N) -> L vertices
    # For L in 22..39:
    #   d = 41-L, center = (d-2) % N = (39-L) % N
    #   The chord is (center, center+d). The long arc goes the other way.
    #   Cycle: center, center-1, center-2, ... going backwards for N-d steps, then chord back.
    #   That's N-d+1 = L vertices.
    
    for L in range(3, 40):
        if L <= 21:
            d = L - 1
            center = (d - 2) % N
            cycle = [(center + i) % N for i in range(L)]
        else:
            d = 41 - L
            center = (d - 2) % N
            # Long arc: go backwards from center
            cycle = [(center - i) % N for i in range(L - 1)] + [center]
            # Wait, let me verify: the chord is (center, center+d). 
            # The long arc from center+d going forward (increasing) to center has N-d edges.
            # So the cycle is: center+d, center+d+1, ..., center (mod N), then chord back to center+d.
            # That's N-d+1 = L vertices.
            start = (center + d) % N
            cycle = [(start + i) % N for i in range(L - 1)] + [start]
        cycles.append(cycle)
    
    # Length 40: Hamiltonian cycle
    cycles.append(list(range(N)))
    
    # Verify validity
    edge_set_check = set((min(a,b), max(a,b)) for a,b in edges)
    valid = True
    for k, cycle in enumerate(cycles, start=3):
        if len(cycle) != k or len(set(cycle)) != k:
            valid = False
            break
        for i in range(k):
            u, v = cycle[i], cycle[(i+1) % k]
            if (min(u,v), max(u,v)) not in edge_set_check:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        cycles = [list(range(k)) for k in range(3, N+1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Try to prune edges
    if elapsed() < DEADLINE - 15:
        edges = _prune(edges, cycles)
    
    return {'edges': edges, 'cycles': cycles}

def _prune(edges, cycles):
    """Remove unnecessary edges while maintaining all cycles."""
    edge_set = set((min(a,b), max(a,b)) for a,b in edges)
    current = list(edge_set)
    
    # Build list of required edges for each cycle
    cycle_edges = []
    for cycle in cycles:
        k = len(cycle)
        ce = set()
        for i in range(k):
            u, v = cycle[i], cycle[(i+1) % k]
            ce.add((min(u,v), max(u,v)))
        cycle_edges.append(ce)
    
    # An edge is needed if it's in at least one cycle's edge set
    # Actually, an edge can be removed if no cycle requires it.
    # But wait - all edges in our construction should be used by at least one cycle.
    # Let's check: try removing each edge and see if all cycles remain valid.
    
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            if elapsed() > DEADLINE - 5:
                break
            trial = current[:i] + current[i+1:]
            trial_set = set(trial)
            # Check all cycles
            ok = True
            for ce in cycle_edges:
                if not ce.issubset(trial_set):
                    ok = False
                    break
            if ok:
                current = trial
                changed = True
                break
    
    return [(a, b) for a, b in current]
