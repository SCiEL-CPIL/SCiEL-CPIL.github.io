import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 30
    # Total edges in K_30
    num_edges = n * (n - 1) // 2
    
    # Precompute lookup table for K_6 subgraphs
    # A K_6 has 15 edges. We represent a coloring as a 15-bit integer.
    # Bit i corresponds to edge i in the K_6.
    # We need to check if the graph defined by the 15 bits contains a K_{2,2,2} subgraph.
    # K_{2,2,2} is a tripartite graph with parts of size 2.
    # It has 12 edges (all between parts).
    # There are 15 ways to partition 6 vertices into 3 pairs.
    # For each partition, if all 12 edges between the pairs are 1, it's a K_{2,2,2}.
    
    # First, define the 15 edge indices in a K_6 (0..5)
    # Edges: (0,1), (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)
    k6_edges = []
    for i in range(6):
        for j in range(i+1, 6):
            k6_edges.append((i, j))
            
    # Map pair (u,v) in 0..5 to bit index
    # We need to check all 15 partitions of {0,1,2,3,4,5} into 3 pairs.
    # Generate all partitions
    def get_partitions():
        partitions = []
        used = [False] * 6
        def backtrack(remaining, current):
            if not remaining:
                partitions.append(tuple(sorted(current)))
                return
            # Pick the smallest unused vertex to avoid duplicates
            first = None
            for i in range(6):
                if not used[i]:
                    first = i
                    break
            if first is None:
                return
            
            # Try pairing first with every other unused vertex
            for j in range(first+1, 6):
                if not used[j]:
                    used[first] = True
                    used[j] = True
                    current.append((first, j))
                    
                    # Find next smallest unused
                    next_first = None
                    for k in range(6):
                        if not used[k]:
                            next_first = k
                            break
                    if next_first is not None:
                        # Recurse on remaining
                        remaining_copy = [k for k in range(6) if not used[k]]
                        # We need to be careful with state. 
                        # Simpler: just generate all 15 perfect matchings manually or recursively
                        pass
                    
                    # This recursive approach is getting messy. Let's just hardcode or generate properly.
                    # Number of perfect matchings in K6 is 15.
                    # Let's just iterate through all combinations of 3 disjoint pairs.
                    pass
            
    # Easier way to generate the 15 matchings:
    # Pairs (a,b), (c,d), (e,f) such that {a,b,c,d,e,f} = {0,1,2,3,4,5}
    matchings = []
    for a in range(6):
        for b in range(a+1, 6):
            remaining = [i for i in range(6) if i != a and i != b]
            for c in range(4):
                for d in range(c+1, 4):
                    e = remaining[c]
                    f = remaining[d]
                    # Check if e,f are the last two
                    # Actually, just check if {e,f} is the set of the other two
                    if len(remaining) == 4:
                        # We picked c, d from 0..3 indices of remaining
                        # This logic is flawed.
                        pass
    
    # Let's just list the 15 matchings explicitly or use a simple generator
    # Matching is a set of 3 pairs.
    # 1: (0,1),(2,3),(4,5)
    # 2: (0,1),(2,4),(3,5)
    # ...
    # We can generate them by permuting 0..5 and taking (p0,p1), (p2,p3), (p4,p5) and normalizing.
    perms = set()
    for p in range(720): # 6!
        # Simple permutation generator
        pass
    
    # Let's use a standard algorithm for perfect matchings in K6
    # There are exactly 15.
    # We can hardcode the edge bitmasks for the 12 edges of each K222.
    # Edge bit index in 15-bit mask:
    # 0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4), 4: (0,5)
    # 5: (1,2), 6: (1,3), 7: (1,4), 8: (1,5)
    # 9: (2,3), 10: (2,4), 11: (2,5)
    # 12: (3,4), 13: (3,5), 14: (4,5)
    
    # For a matching (pairs P1, P2, P3), the edges of K222 are all edges between Pi and Pj for i!=j.
    # There are 3*3 = 9? No, 3 pairs. Edges between P1 and P2 (4 edges), P1 and P3 (4 edges), P2 and P3 (4 edges). Total 12 edges.
    
    # Let's generate the 12-edge masks for the 15 matchings
    k222_masks = []
    # Generate all 15 perfect matchings
    # We can do this by fixing 0's partner, then 2's partner, etc.
    # 0 is paired with 1, 2, 3, 4, 5.
    
    # Helper to get bit index of edge (u,v) in K6
    def edge_bit_idx(u, v):
        if u > v: u, v = v, u
        # (0,1)->0, (0,2)->1, (0,3)->2, (0,4)->3, (0,5)->4
        # (1,2)->5, (1,3)->6, (1,4)->7, (1,5)->8
        # (2,3)->9, (2,4)->10, (2,5)->11
        # (3,4)->12, (3,5)->13, (4,5)->14
        if u == 0:
            return v - 1
        if u == 1:
            return 5 + (v - 2) # v in 2..5 -> 0..3 -> 5..8
        if u == 2:
            return 9 + (v - 3) # v in 3..5 -> 0..2 -> 9..11
        if u == 3:
            return 12 + (v - 4) # v in 4..5 -> 0..1 -> 12..13
        if u == 4:
            return 14 # v must be 5
    
    # Generate matchings
    # 0-1
    # 2-3, 4-5
    # 2-4, 3-5
    # 2-5, 3-4
    # 0-2
    # 1-3, 4-5
    # 1-4, 3-5
    # 1-5, 3-4
    # 0-3
    # 1-2, 4-5
    # 1-4, 2-5
    # 1-5, 2-4
    # 0-4
    # 1-2, 3-5
    # 1-3, 2-5
    # 1-5, 2-3
    # 0-5
    # 1-2, 3-4
    # 1-3, 2-4
    # 1-4, 2-3
    
    matchings_list = [
        [(0,1), (2,3), (4,5)],
        [(0,1), (2,4), (3,5)],
        [(0,1), (2,5), (3,4)],
        [(0,2), (1,3), (4,5)],
        [(0,2), (1,4), (3,5)],
        [(0,2), (1,5), (3,4)],
        [(0,3), (1,2), (4,5)],
        [(0,3), (1,4), (2,5)],
        [(0,3), (1,5), (2,4)],
        [(0,4), (1,2), (3,5)],
        [(0,4), (1,3), (2,5)],
        [(0,4), (1,5), (2,3)],
        [(0,5), (1,2), (3,4)],
        [(0,5), (1,3), (2,4)],
        [(0,5), (1,4), (2,3)]
    ]
    
    for m in matchings_list:
        mask = 0
        # Pairs are m[0], m[1], m[2]
        # Edges between m[0] and m[1]
        for u in m[0]:
            for v in m[1]:
                mask |= (1 << edge_bit_idx(u, v))
        # Edges between m[0] and m[2]
        for u in m[0]:
            for v in m[2]:
                mask |= (1 << edge_bit_idx(u, v))
        # Edges between m[1] and m[2]
        for u in m[1]:
            for v in m[2]:
                mask |= (1 << edge_bit_idx(u, v))
        k222_masks.append(mask)
        
    # Precompute lookup table
    # lookup[i] = 1 if graph defined by i contains K222, else 0
    lookup = [0] * (1 << 15)
    for i in range(1 << 15):
        for kmask in k222_masks:
            if (i & kmask) == kmask:
                lookup[i] = 1
                break
                
    # Precompute edge indices for all 6-subsets of 30 vertices
    # This might take some memory/time, but 594k subsets * 15 edges is ~9M integers.
    # We can generate them on the fly or precompute a list of (subset, edge_indices)
    # To save time, we precompute the list of 6-subsets and their corresponding edge indices in the global graph.
    
    # Global edge index for (i,j) where i<j
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        # i is 0..29
        # Number of edges before i: i*(i-1)/2
        # Offset: j - i - 1
        return i * (i - 1) // 2 + (j - i - 1)
        
    # Precompute all 6-subsets and their edge index lists
    subsets_data = []
    for verts in combinations(range(n), 6):
        # Map local indices 0..5 to global indices
        # verts is sorted tuple
        edge_indices = []
        for local_u in range(6):
            for local_v in range(local_u + 1, 6):
                global_u = verts[local_u]
                global_v = verts[local_v]
                edge_indices.append(get_edge_idx(global_u, global_v))
        subsets_data.append(edge_indices)
        
    # Initial state: Random or Paley-based?
    # Let's start with a random state
    current_colors = [random.randint(0, 1) for _ in range(num_edges)]
    
    def get_cost(colors):
        # colors is a list of 0/1
        cost = 0
        for edge_indices in subsets_data:
            # Build 15-bit mask for this subset
            # Bit k in mask corresponds to edge_indices[k]
            # We want to check if the subgraph in Color 0 has K222 and Color 1 has K222
            # Mask for Color 0: 1 if color is 0
            # Mask for Color 1: 1 if color is 1
            
            mask0 = 0
            mask1 = 0
            for k, e_idx in enumerate(edge_indices):
                c = colors[e_idx]
                if c == 0:
                    mask0 |= (1 << k)
                else:
                    mask1 |= (1 << k)
            
            cost += lookup[mask0] + lookup[mask1]
        return cost

    current_cost = get_cost(current_colors)
    best_colors = current_colors[:]
    best_cost = current_cost
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Simulated Annealing
    while T > T_min and time.time() < deadline:
        # Pick a random edge to flip
        e_idx = random.randint(0, num_edges - 1)
        old_color = current_colors[e_idx]
        new_color = 1 - old_color
        current_colors[e_idx] = new_color
        
        new_cost = get_cost(current_colors)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = current_colors[:]
        else:
            # Revert
            current_colors[e_idx] = old_color
            
        T *= cooling
        
    # Verify best solution
    # If best_cost == 0, we found a valid 30-vertex coloring
    if best_cost == 0:
        return {'n': 30, 'colors': best_colors}
    else:
        # Fallback to 29 (Paley)
        n_base = 29
        exponent = (n_base - 1) // 2
        adj = [[0] * n_base for _ in range(n_base)]
        for i in range(n_base):
            for j in range(i + 1, n_base):
                diff = (j - i) % n_base
                res = pow(diff, exponent, n_base)
                if res == 1:
                    adj[i][j] = 1
                    adj[j][i] = 1
                else:
                    adj[i][j] = 0
                    adj[j][i] = 0
        colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        return {'n': 29, 'colors': colors}
