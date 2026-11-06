"""Construct a sparse pancyclic graph on 40 vertices with minimum edges."""
import sys

N = 40

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a pancyclic graph on 40 vertices with minimum edge count.
    
    Strategy: Use a base cycle (0-1-2-...-39-0) plus a fan of chords from vertex 0
    to vertices 2 through 20. This gives 40 + 19 = 59 edges.
    
    For each cycle length L (3 to 40), we construct a simple cycle:
    - L = 40: the base cycle itself
    - L = 39: remove one vertex from base cycle, use two base edges and one chord
    - L = 38: similar
    - ...
    - For various L, use chords to "skip" vertices appropriately
    
    The key insight: with chords (0,k) for k=2..20, we can form cycles of any length
    3..40 by appropriate combinations of base path segments and chords.
    
    Specifically, for a cycle of length L:
    - If L >= 21: Use the "long arc" - go from 0 clockwise around the cycle skipping
      a contiguous block, then use chord to jump back
    - If L <= 19: Use the "short arc" - use chord (0, 40-L) to jump, then traverse
      the remaining path
    - Special cases for small L
    """
    
    # Base cycle edges
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add chords from 0 to vertices 2 through 20
    for k in range(2, 21):
        edges.add((0, k))
    
    # Convert to list of sorted tuples
    edge_list = [tuple(sorted(e)) for e in edges]
    edge_list.sort()
    
    # Edge lookup for validation
    edge_set = set(edge_list)
    
    # Build cycles for each length 3..40
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:  # 40
            # Base cycle
            cyc = list(range(N))
        elif L == N - 1:  # 39
            # Remove vertex 1, use chord (0,2) instead of (0,1) and (1,2)
            # Path: 0, 39, 38, ..., 2, 0
            # Wait, that's not a simple cycle of length 39
            # Actually: 0, 39, 38, ..., 2, then chord (2,0)
            # That gives vertices 0, 39, 38, ..., 2 = 39 vertices
            # Edges: (0,39), (39,38), ..., (3,2), (2,0) - all valid
            cyc = [0] + list(range(N - 1, 1, -1))  # 0, 39, 38, ..., 2
        elif L >= 21:
            # d = N - L, use long arc skipping d consecutive vertices
            # Chord (0, d) where d = N - L
            # If d == 1, L = 39, handled above
            d = N - L
            if d == 1:
                cyc = [0] + list(range(N - 1, 0, -1))  # 0, 39, ..., 1
            elif d <= 20:
                # Use chord (0, d), traverse long way around
                # 0, 39, 38, ..., d, 0
                cyc = [0] + list(range(N - 1, d - 1, -1))
            else:
                # d > 20, need different approach
                # For L = 20, d = 20, handled above (d=20 <= 20)
                # For L = 19, d = 21, use short arc instead
                # This shouldn't happen for L >= 21 since d <= 19
                cyc = list(range(L))  # fallback
        else:  # L <= 20
            # Use chord (0, L-1) or similar
            # For L=3: triangle 0-1-2-0 uses (0,1),(1,2),(2,0) - chord (0,2) exists
            # For L=4: 0-1-2-3-0 uses (0,1),(1,2),(2,3),(3,0) - (3,0) is chord, exists
            # General: use vertices 0, 1, 2, ..., L-1
            # Edges: (0,1),(1,2),...,(L-2,L-1),(L-1,0)
            # (L-1,0) is a chord if L-1 >= 2, which is true for L >= 3
            # And L-1 <= 19 for L <= 20, so chord exists
            cyc = list(range(L))
        
        # Validate the cycle
        assert len(cyc) == L, f"Cycle length {len(cyc)} != {L}"
        assert len(set(cyc)) == L, f"Duplicate vertices in cycle of length {L}"
        for i in range(L):
            u = cyc[i]
            v = cyc[(i + 1) % L]
            e = (min(u, v), max(u, v))
            assert e in edge_set, f"Edge {e} not in edge set for cycle length {L}"
        
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}

# EVOLVE-BLOCK-END
