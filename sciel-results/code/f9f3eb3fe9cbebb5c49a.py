import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # We target n=17
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Precompute all C6 cycles
    # A C6 is defined by 6 distinct vertices v0..v5
    # Edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
    # We need to map these to edge indices in the lexicographic list
    # Edge index for (i,j) with i<j is: i*n - i*(i+1)//2 + (j-i-1)
    # Formula: idx(i, j) = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # Or simpler: 
    # row i has n-1-i edges.
    # offset[i] = sum_{k=0}^{i-1} (n-1-k)
    
    def edge_idx(i, j):
        if i > j: i, j = j, i
        # Number of edges before row i
        # Row 0: n-1
        # Row 1: n-2
        # ...
        # Row i-1: n-i
        # Sum = (n-1) + (n-2) + ... + (n-i)
        # This is i*n - i*(i+1)//2
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    cycles = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    # Generate all combinations of 6 vertices
    # There are C(17, 6) = 12376 combinations
    # For each combination, there are 60 distinct cycles (6! / 12)
    # Actually, number of cycles in K6 is 60.
    # We can just iterate permutations of 6 vertices and normalize to avoid duplicates?
    # Better: iterate combinations, then generate all 60 cycles for that set.
    
    import itertools
    
    vertices = range(n)
    cycle_count = 0
    for comb in itertools.combinations(vertices, 6):
        # Generate all 60 cycles on these 6 vertices
        # Fix v0 = comb[0] to reduce symmetry?
        # A cycle is a permutation of the 6 vertices up to rotation and reversal.
        # Number of distinct cycles on 6 labeled vertices is 60.
        # We can generate them by fixing the first vertex and permuting the other 5,
        # then dividing by 2 for reversal?
        # Easier: iterate all permutations of the 6 vertices, take the first 6,
        # and use a set to store normalized tuple of edges to avoid duplicates.
        
        # But 12376 * 720 permutations is ~9 million.
        # Storing edge sets in a set is okay.
        
        # Actually, we can just generate the 60 cycles explicitly.
        # Let the vertices be v0, v1, v2, v3, v4, v5.
        # Fix v0. The remaining 5 can be arranged in 5! / 2 = 60 ways.
        # We fix v0 = comb[0].
        # Permute comb[1:]
        # For each perm, the cycle is v0, p0, p1, p2, p3, p4, v0.
        # To avoid reversal duplicates, we can require p0 < p4?
        # Or just use a set of frozensets of edges.
        
        seen_cycles_in_comb = set()
        for perm in itertools.permutations(comb[1:]):
            cyc_verts = (comb[0],) + perm
            # Edges
            edges = []
            valid = True
            for k in range(6):
                u = cyc_verts[k]
                v = cyc_verts[(k + 1) % 6]
                e = edge_idx(u, v)
                edges.append(e)
            
            # Normalize edge set to check for duplicates (reversal/rotation)
            # Since vertices are fixed in comb, rotation is handled by fixing comb[0].
            # Reversal: (v0, v1, v2, v3, v4, v5) is same as (v0, v5, v4, v3, v2, v1).
            # So we can just check if the set of edges is already seen.
            edge_set = frozenset(edges)
            if edge_set not in seen_cycles_in_comb:
                seen_cycles_in_comb.add(edge_set)
                cycles.append(edges)
                for e in edges:
                    edge_to_cycles[e].append(len(cycles) - 1)
                cycle_count += 1

    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Function to check if a cycle is monochromatic
    def is_mono(cyc_idx):
        c = cycles[cyc_idx]
        col = colors[c[0]]
        for e in c[1:]:
            if colors[e] != col:
                return False
        return True

    # Calculate initial cost
    cost = 0
    for i in range(len(cycles)):
        if is_mono(i):
            cost += 1
            
    best_cost = cost
    best_colors = colors[:]
    
    # If we start with 0 cost, return immediately
    if cost == 0:
        return {'n': n, 'colors': colors}

    # Local Search
    # Precompute list of edges involved in mono cycles for faster selection
    # We can just pick random edge and check if it's in a mono cycle
    
    while time.time() < deadline:
        if cost == 0:
            break
            
        # Pick an edge
        # Strategy: Pick an edge that is in at least one mono cycle
        # If we don't track which edges are in mono cycles, we can just pick random
        # and see if it helps.
        # To be more directed, let's maintain a list of "bad" edges?
        # That's expensive to update.
        # Let's just pick a random edge.
        
        e = random.randint(0, num_edges - 1)
        
        old_color = colors[e]
        
        best_new_color = old_color
        best_delta = 0
        
        for new_color in range(4):
            if new_color == old_color:
                continue
                
            # Calculate delta
            delta = 0
            for cyc_idx in edge_to_cycles[e]:
                # Check if cycle was mono before
                was_mono = True
                # We know colors[e] is old_color
                # We need to check other 5 edges
                c = cycles[cyc_idx]
                # Optimization: if old_color != new_color, and the cycle was mono with old_color,
                # it is no longer mono.
                # If the cycle was not mono, it might become mono with new_color.
                
                # Check if all other edges have old_color
                # If yes, it was mono.
                all_old = True
                for ee in c:
                    if ee == e: continue
                    if colors[ee] != old_color:
                        all_old = False
                        break
                if all_old:
                    # It was mono. Now e is new_color, so it's not mono.
                    delta -= 1
                    
                # Check if all other edges have new_color
                # If yes, and e becomes new_color, it becomes mono.
                all_new = True
                for ee in c:
                    if ee == e: continue
                    if colors[ee] != new_color:
                        all_new = False
                        break
                if all_new:
                    # It will become mono.
                    delta += 1
                    
            if delta < best_delta:
                best_delta = delta
                best_new_color = new_color
                
        # Apply best change
        if best_new_color != old_color:
            colors[e] = best_new_color
            cost += best_delta
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
                
        # If we have a valid solution, we can return
        if cost == 0:
            break
            
    # If we found a valid solution for n=17, return it
    if best_cost == 0:
        return {'n': n, 'colors': best_colors}
        
    # If not, fall back to seed n=9
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
