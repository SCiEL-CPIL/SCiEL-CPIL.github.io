"""
Constructs a sparse pancyclic graph on 40 vertices.
Uses a two-hub structure with a central connection to ensure 2-connectivity
and allow Hamiltonian cycles, while using spokes to generate all intermediate
cycle lengths.
"""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    
    # Define two hubs
    hub1 = 0
    hub2 = 20
    
    # Segment 1: Vertices 1 to 19
    seg1 = list(range(1, 20))
    # Segment 2: Vertices 21 to 39
    seg2 = list(range(21, 40))
    
    # Add spokes for Hub 1
    for v in seg1:
        edges.append((hub1, v))
        
    # Add spokes for Hub 2
    for v in seg2:
        edges.append((hub2, v))
        
    # Add path edges for Segment 1
    for i in range(len(seg1) - 1):
        edges.append((seg1[i], seg1[i+1]))
        
    # Add path edges for Segment 2
    for i in range(len(seg2) - 1):
        edges.append((seg2[i], seg2[i+1]))
        
    # Connect the two segments to ensure 2-connectivity and allow long cycles
    # Bridge between the ends of the paths
    edges.append((seg1[-1], seg2[0]))  # 19 - 21
    # Connect the hubs to allow cycles spanning both segments
    edges.append((hub1, hub2))        # 0 - 20
    
    # Generate cycles
    # The graph contains a Hamiltonian cycle:
    # 0-1-2-...-19-21-22-...-39-20-0
    # By using the spokes (0-i and 20-j), we can form cycles of all lengths.
    
    cycles = []
    
    # Helper to extract a cycle of length k
    # Strategy:
    # For small k (3..20), use the fan structure around Hub 1.
    # Cycle: 0 - 1 - 2 - ... - (k-1) - 0
    for k in range(3, 21):
        # Vertices: 0, 1, ..., k-1
        # Edges: (0,1), (1,2)...(k-2, k-1), (k-1, 0)
        # Note: (k-1, 0) is a spoke if k-1 is in seg1.
        # For k=20, vertices 0..19. (19,0) is a spoke.
        cycles.append([0] + list(range(1, k)))
        
    # For large k (21..40), we need to involve Hub 2.
    # A generic Hamiltonian cycle is:
    # 0 - 1 - ... - 19 - 21 - 22 - ... - 39 - 20 - 0
    # Length 40.
    # To get length k < 40, we can "shortcut" the path using spokes.
    # For example, to get length 39:
    # Replace 0-1-2-...-19-21-...-39-20-0
    # We can jump from 1 to 20? No, no edge.
    # We can jump from 19 to 20? No.
    # We can jump from 1 to 0? That's the end.
    
    # Let's construct cycles explicitly for k=21..40.
    # The base path is P = [0, 1, 2, ..., 19, 21, 22, ..., 39, 20]
    # This path has 40 vertices.
    # The cycle is P + [0].
    # To reduce the length, we can replace a subpath with a shorter connection if available.
    # Available shortcuts:
    # - Any (0, i) for i in 1..19
    # - Any (20, j) for j in 21..39
    # - (0, 20)
    # - (19, 21)
    
    # For k in 21..39:
    # We can form a cycle by taking a prefix of the Hamiltonian path and a suffix,
    # and closing it with a chord.
    
    # Example for k=39:
    # Path: 0-1-...-19-21-...-38-39-20-0
    # If we use chord (1, 20)? No edge.
    # If we use chord (0, 20)?
    # Cycle: 0-1-...-39-20-0.
    # Length: 1 (0-1) + 18 (1-19) + 1 (19-21) + 18 (21-39) + 1 (39-20) + 1 (20-0) = 40.
    # This is the full cycle.
    
    # To get 39:
    # Remove vertex 1?
    # Cycle: 0-2-3-...-19-21-...-39-20-0?
    # Edges: 0-2 (spoke), 2-3...19 (path), 19-21 (bridge), 21-...-39 (path), 39-20 (spoke), 20-0 (hub).
    # Vertices: 0, 2..19 (18), 21..39 (19), 20. Total 1+18+19+1 = 39.
    # Yes!
    
    # General pattern for k in 21..39:
    # Let m = 40 - k. We need to remove m vertices from the Hamiltonian cycle.
    # We can remove vertices from the "middle" of the path segments by using spokes to jump over them.
    # Specifically, we can jump from 0 to i (skipping 1..i-1) or from 20 to j (skipping j+1..39).
    
    # Let's construct cycles for 21..40.
    # k=40: [0, 1, ..., 19, 21, ..., 39, 20]
    # k=39: Remove 1. Cycle: 0-2-...-19-21-...-39-20-0.
    #         Uses (0,2).
    # k=38: Remove 1,2. Cycle: 0-3-...-19-21-...-39-20-0.
    #         Uses (0,3).
    # ...
    # k=21: Remove 1..18. Cycle: 0-19-21-...-39-20-0.
    #         Uses (0,19).
    #         Vertices: 0, 19, 21..39, 20. Count: 1 + 1 + 19 + 1 = 22?
    #         Wait. 21..39 is 19 vertices.
    #         0, 19, 21, 22, ..., 39, 20.
    #         Count: 1 (0) + 1 (19) + 19 (21-39) + 1 (20) = 22.
    #         So this gives length 22.
    #         We need length 21.
    #         To get 21, remove one more.
    #         Remove 39?
    #         Cycle: 0-19-21-...-38-20-0.
    #         Uses (0,19) and (20,38)? No, (38,20) is a spoke.
    #         Vertices: 0, 19, 21..38, 20.
    #         Count: 1 + 1 + 18 + 1 = 21.
    #         Yes.
    
    # So for k in 22..40, we can use prefix shortcuts from 0.
    # For k in 21..22, we might need to mix or use suffix shortcuts.
    
    cycles_large = []
    # k=40 down to 22
    for k in range(40, 21, -1):
        # Number of vertices to skip from the start (after 0)
        # Path: 0 - 1 - ... - 19
        # If we jump 0 -> i, we skip 1..i-1.
        # Let skip_start = i - 1.
        # The cycle is 0 - i - ... - 19 - 21 - ... - 39 - 20 - 0.
        # Length = 1 (0-i) + (19 - i + 1) (i..19) + 1 (19-21) + 19 (21..39) + 1 (39-20) + 1 (20-0)
        # Length = 1 + (20 - i) + 1 + 19 + 1 + 1 = 43 - i.
        # We want Length = k.
        # 43 - i = k => i = 43 - k.
        # Since i must be in 1..19.
        # If k=40, i=3. Cycle 0-3-...-19-21-...-39-20-0.
        # Vertices: 0, 3..19 (17), 21..39 (19), 20. Total 1+17+19+1 = 38?
        # Wait.
        # My length calculation:
        # 0-i (1 edge)
        # i..19 (19-i edges? No, vertices i, i+1, ..., 19. Count 19-i+1. Edges 19-i.)
        # 19-21 (1 edge)
        # 21..39 (18 edges? Vertices 21..39 count 19. Edges 18.)
        # 39-20 (1 edge)
        # 20-0 (1 edge)
        # Total edges: 1 + (19-i) + 1 + 18 + 1 + 1 = 41 - i.
        # We want k edges.
        # k = 41 - i => i = 41 - k.
        
        i = 41 - k
        if 1 <= i <= 19:
            # Construct cycle
            # 0, i, i+1, ..., 19, 21, 22, ..., 39, 20
            cyc = [0]
            cyc.extend(range(i, 20))
            cyc.extend(range(21, 40))
            cyc.append(20)
            cycles_large.append(cyc)
            
    # We generated cycles for k where 1 <= 41-k <= 19 => 22 <= k <= 40.
    # So we have 22..40.
    # We need 21.
    # For k=21:
    # i = 41 - 21 = 20. Not in 1..19.
    # We need to skip from the other side too.
    # Skip 1..18 (i=19) and skip 39 (j=38? No, skip 39 means end at 38).
    # Cycle: 0-19-21-...-38-20-0.
    # Edges: 0-19 (1), 19-21 (1), 21-38 (17), 38-20 (1), 20-0 (1).
    # Total: 1+1+17+1+1 = 21.
    # Vertices: 0, 19, 21..38, 20. Count 1+1+18+1 = 21.
    cycles_large.append([0, 19] + list(range(21, 39)) + [20])
    
    # We have cycles for 3..20 (small) and 21..40 (large).
    # Combine and sort by length? The task requires "one vertex-list cycle of every length 3..40 in that order".
    
    all_cycles = cycles + cycles_large
    
    # Re-order to 3..40
    # small cycles are 3..20
    # large cycles are 22..40 and 21
    # Let's just map them properly.
    
    final_cycles = []
    for k in range(3, 41):
        if k <= 20:
            # From 'cycles' list
            # cycles[0] is k=3, ..., cycles[17] is k=20
            final_cycles.append(cycles[k-3])
        else:
            # From 'cycles_large'
            # We need to find the cycle of length k
            # It's easier to regenerate or store in dict
            pass
            
    # Let's refactor to be clean.
    
    edges_set = set()
    for e in edges:
        edges_set.add((min(e), max(e)))
        
    def get_cycle(k):
        if k <= 20:
            return [0] + list(range(1, k))
        elif k == 21:
            return [0, 19] + list(range(21, 39)) + [20]
        else:
            i = 41 - k
            # i is in 1..19
            return [0] + list(range(i, 20)) + list(range(21, 40)) + [20]
            
    final_cycles = [get_cycle(k) for k in range(3, 41)]
            
    return {'edges': edges, 'cycles': final_cycles}

# EVOLVE-BLOCK-END
