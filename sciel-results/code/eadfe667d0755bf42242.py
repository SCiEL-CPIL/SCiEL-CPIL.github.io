"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Pre-computed 59-edge solution: Hamiltonian cycle on 40 vertices
    # plus 19 chords from vertex 0 to vertices 2..20.
    #
    # Edge count: 40 (cycle) + 19 (chords) = 59
    # Score: 59 - 40 = 19 (lower is better)
    #
    # Cycle certificate for each length L in 3..40:
    # - L=40: the full cycle [0,1,2,...,39]
    # - L=3..20: chord (0, L-1) gives cycle [0, 1, 2, ..., L-1]
    #   (uses edges (0,1), (1,2), ..., (L-2,L-1), (L-1,0))
    #   Wait, that's not right. Let me reconsider.
    #
    # The chord (0,k) in the cycle 0-1-2-...-39-0 creates two cycles:
    #   - Short cycle: 0-1-2-...-k-0, length k+1
    #   - Long cycle: 0-k-(k+1)-...-39-0, length 40-(k+1)+1 = 40-k
    #
    # For k=2: cycles of length 3 and 38
    # For k=3: cycles of length 4 and 37
    # ...
    # For k=19: cycles of length 20 and 22
    # For k=20: cycle of length 21 (both halves are length 21)
    #
    # So chords (0,2) through (0,20) cover lengths: 3,38, 4,37, 5,36, ..., 19,23, 20,22, 21
    # That's lengths 3..21 and 22..38, i.e., 3..38. Plus the base cycle gives 40.
    # Missing: 39.
    #
    # Wait, let me recheck. k ranges from 2 to 20.
    # k=2: lengths 3 and 39 (not 38!)
    #   Short: 0-1-2-0, length 3
    #   Long: 0-2-3-...-39-0, length 39
    # k=3: lengths 4 and 38
    # k=4: lengths 5 and 37
    # ...
    # k=19: lengths 20 and 22
    # k=20: length 21
    #
    # So covered: 3,39, 4,38, 5,37, ..., 19,23, 20,22, 21
    # That's 3..21 and 22..39, i.e., all of 3..39. Plus 40 from base cycle.
    # Total: 3..40. ✓
    
    # Build edges
    cycle_edges = [(i, (i+1) % N) for i in range(N)]
    chords = [(0, k) for k in range(2, 21)]
    all_edges = cycle_edges + chords
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Build cycle certificates
    cycles = []
    for length in range(3, N+1):
        if length == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        elif length <= 20:
            # Use chord (0, length-1): short cycle 0-1-2-...-(length-1)-0
            k = length - 1
            cycles.append(list(range(k + 1)))
        elif length == 21:
            # Chord (0,20): cycle 0-1-2-...-20-0
            cycles.append(list(range(21)))
        else:
            # length is 22..39, use chord (0, k) where long cycle has this length
            # Long cycle length = N - (k+1) + 1 = N - k
            # So N - k = length => k = N - length
            k = N - length
            # Long cycle: 0-k-(k+1)-...-39-0
            # This visits: 0, k, k+1, k+2, ..., 39
            # Length: 1 + (39-k) + 1 = 40-k = length ✓
            # But we need distinct vertices in a cycle.
            # The cycle is: 0, k, k+1, ..., 39, back to 0.
            # That's vertices 0, k, k+1, ..., 39.
            # Count: 1 + (39 - k + 1) = 40 - k = length ✓
            # All distinct since k >= 2 (length <= 38) and 0 is not in [k..39] when k>=1.
            # For length=39: k=1, but k=1 is not in our chords!
            # 
            # Wait, for length=39: k = 40-39 = 1. But chord (0,1) is part of the cycle, not a chord.
            # The "long cycle" from chord (0,2) has length 39: 0-2-3-...-39-0.
            # So for length=39, we use k=2.
            #
            # Let me recheck: for chord (0,k), long cycle length = N - k.
            # length = N - k => k = N - length.
            # length=39: k=1. But we don't have chord (0,1).
            # length=38: k=2. Chord (0,2) exists. Long cycle: 0-2-3-...-39-0, length 38. ✓
            #
            # So length=39 must come from chord (0,2)'s... wait no.
            # Chord (0,2): short=3, long=39? Let's recount.
            # 0-2-3-4-...-39-0: vertices are 0,2,3,...,39. Count = 1+38=39. ✓
            # So long cycle of chord (0,2) has length 39.
            # N - k = 40 - 2 = 38. That's wrong!
            #
            # I made an error. Let me recount.
            # Cycle is 0-1-2-...-39-0 (40 edges, 40 vertices).
            # Chord (0,k) adds edge (0,k).
            # Short cycle: 0-1-2-...-k-0. Vertices: 0,1,...,k. Count: k+1. Edges: (0,1),(1,2),...,(k-1,k),(k,0). Count: k. 
            # Wait, k+1 vertices in a cycle means k+1 edges.
            # Vertices: 0,1,2,...,k. That's k+1 vertices. Edges: (0,1),(1,2),...,(k-1,k),(k,0). That's k edges + 1 chord = k+1 edges? No.
            # Edges used: (0,1), (1,2), ..., (k-1,k), (k,0). The last edge (k,0) is the chord.
            # Number of cycle edges: k-1 consecutive cycle edges + 1 chord = k edges for k+1 vertices? No.
            # A cycle with m vertices has m edges.
            # Vertices: 0,1,...,k → m = k+1.
            # Edges: (0,1),(1,2),...,(k-1,k) are k-1 cycle edges, plus (k,0) chord = k edges total? That's only k edges for k+1 vertices. That's not a cycle.
            #
            # I'm confusing myself. Let me be very careful.
            # The cycle 0-1-2-...-k-0 uses vertices {0,1,...,k} and edges {(0,1),(1,2),...,(k-1,k),(k,0)}.
            # Number of vertices: k+1.
            # Number of edges: k. 
            # That's not a valid cycle! A cycle with m vertices must have m edges.
            #
            # Ah, I see the issue. The edge (k,0) is the chord. The edges (0,1),(1,2),...,(k-1,k) are k-1 edges from the base cycle.
            # Total edges in this "cycle": (k-1) + 1 = k edges. But we have k+1 vertices.
            # This is NOT a cycle. It's a path from 0 to k with an extra edge (k,0) making it a cycle of length... 
            # Actually, 0-1-2-...-k-0 IS a cycle. It has k+1 vertices and k+1 edges? No, it has k+1 vertices and the edges are (0,1),(1,2),...,(k-1,k),(k,0). That's k edges.
            # 
            # Wait: from 0 to k via 1,2,...,k-1: that's k steps (edges). Then from k back to 0: 1 edge. Total: k+1 edges? No.
            # 0→1 (1), 1→2 (2), ..., (k-1)→k (k). That's k edges to go from 0 to k. Then k→0 is edge k+1.
            # So the cycle has k+1 edges and k+1 vertices. ✓
            # 
            # I was miscounting. The consecutive edges from 0 to k along the base cycle are: (0,1), (1,2), ..., (k-1,k). That's k edges.
            # Plus the chord (k,0). Total: k+1 edges. ✓
            #
            # So short cycle length = k+1.
            # Long cycle: 0-k-(k+1)-...-39-0.
            # Vertices: 0, k, k+1, ..., 39. Count: 1 + (39-k) = 40-k.
            # Edges: chord (0,k), then cycle edges (k,k+1), ..., (38,39), (39,0). Count: 1 + (39-k) + 1 = 41-k? 
            # From k to 39 along base cycle: (k,k+1), ..., (38,39). That's 39-k edges.
            # Plus (39,0) and (0,k). Total: 1 + (39-k) + 1 = 41-k. But vertices are 40-k.
            #
            # 40-k vertices should have 40-k edges.
            # Let me recount. Vertices in long cycle: 0, k, k+1, ..., 39.
            # That's vertices {0} ∪ {k, k+1, ..., 39}. Size: 1 + (39-k+1) = 40-k. ✓
            # Edges: (0,k) chord, (k,k+1), (k+1,k+2), ..., (38,39), (39,0).
            # Consecutive edges from k to 39: (k,k+1), ..., (38,39). Count: 39-k.
            # Plus (39,0) and (0,k). Total: (39-k) + 2 = 41-k.
            # But 40-k vertices need 40-k edges. 41-k ≠ 40-k.
            #
            # I think I'm double-counting or miscounting. Let me trace for a small example.
            # N=6, cycle 0-1-2-3-4-5-0. Chord (0,2).
            # Short: 0-1-2-0. Vertices {0,1,2}, 3 vertices. Edges: (0,1),(1,2),(2,0). 3 edges. ✓
            # Long: 0-2-3-4-5-0. Vertices {0,2,3,4,5}, 5 vertices. Edges: (0,2),(2,3),(3,4),(4,5),(5,0). 5 edges. ✓
            # N-k = 6-2 = 4. But long cycle has 5 vertices!
            #
            # So my formula was wrong. Long cycle length = N - k + 1? 6-2+1=5. ✓
            # Short cycle length = k + 1. 2+1=3. ✓
            # Sum: (k+1) + (N-k+1) = N+2. But the two cycles share the chord (0,k), so total distinct edges = N + 1 (base N + 1 chord). ✓
            #
            # So for N=40:
            # Chord (0,k): short length = k+1, long length = 40-k+1 = 41-k.
            # k=2: short=3, long=39
            # k=3: short=4, long=38
            # ...
            # k=19: short=20, long=22
            # k=20: short=21, long=21
            #
            # Covered lengths: 3,39, 4,38, ..., 20,22, 21.
            # That's 3..21 ∪ 22..39 = 3..39. Plus base cycle 40. ✓
            
    # Fix the cycle construction
    cycles = []
    for length in range(3, N+1):
        if length == N:
            cycles.append(list(range(N)))
        elif length <= 21:
            # Short cycle from chord (0, length-1)
            k = length - 1
            cycles.append(list(range(k + 1)))
        else:
            # Long cycle from chord (0, k) where 41-k = length => k = 41-length
            k = 41 - length
            # Cycle: 0, k, k+1, ..., 39
            cyc = [0] + list(range(k, N))
            cycles.append(cyc)
    
    # Verify all cycles
    for i, cyc in enumerate(cycles):
        length = i + 3
        if len(cyc) != length or len(set(cyc)) != length:
            raise RuntimeError(f"Invalid cycle for length {length}")
        for j in range(len(cyc)):
            u, v = cyc[j], cyc[(j+1) % len(cyc)]
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                raise RuntimeError(f"Missing edge {e} in cycle for length {length}")
    
    return {'edges': [list(e) for e in sorted(edge_set)], 'cycles': cycles}

# EVOLVE-BLOCK-END
