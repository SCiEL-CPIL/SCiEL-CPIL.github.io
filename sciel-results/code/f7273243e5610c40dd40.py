import time
import random
import sys

def construct():
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    # The evaluator kills at 150s. We aim for 140s safety margin.
    total_time_limit = 140.0
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        if remaining_time < 2.0:
            # Fallback to trivial partition (edges) if almost out of time
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time: first instance is smaller, second is larger
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.3, 30.0)
            else:
                instance_time_limit = min(remaining_time * 0.7, 100.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
        
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def solve_instance(n, p, seed, time_limit):
    # Generate edges and build bitset adjacency
    rng = random.Random(seed)
    adj = [0] * n
    edge_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
    if edge_count == 0:
        return []
        
    cliques = []
    start_time = time.time()
    
    # Mask for all vertices
    all_mask = (1 << n) - 1
    
    # We maintain a list of active vertices? 
    # Actually, we can just work with the global adj and remove edges.
    # To speed up max clique finding, we should restrict the search space 
    # to vertices that still have edges.
    
    while edge_count > 0:
        if time.time() - start_time > time_limit:
            # Fallback to edges for remaining
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        edge_count -= 1
                    mask ^= lsb
            break
            
        # Find max clique in current adj
        # We can pass a mask of vertices that are "active" (have degree > 0)
        # to prune the search space.
        active_mask = 0
        for u in range(n):
            if adj[u] != 0:
                active_mask |= (1 << u)
        
        clique = find_max_clique(adj, n, active_mask)
        
        if not clique:
            # Safety: if no clique found but edges exist
            # Pick any edge
            found = False
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                    mask ^= lsb
                if found: break
            if not found: break
            
        # Remove edges of clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edge_count -= 1
                    
        cliques.append(clique)
        
    return cliques

def find_max_clique(adj, n, init_mask):
    best = []
    best_len = 0
    
    # Recursive Bron-Kerbosch with coloring
    # C: current clique
    # P: candidate vertices (bitmask)
    # X: excluded vertices (bitmask) - not strictly needed if we just want max clique
    # and we don't care about enumerating all, just the largest.
    # Standard Max Clique algorithm:
    # expand(C, P)
    
    def expand(C, P):
        nonlocal best, best_len
        
        if not P:
            if len(C) > best_len:
                best_len = len(C)
                best = C[:]
            return
            
        # Bound: len(C) + popcount(P)
        if len(C) + P.bit_count() <= best_len:
            return
            
        # Get vertices in P
        verts = []
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            verts.append(v)
            temp_P ^= lsb
            
        # Greedy Coloring of P
        # We color vertices in P to get an upper bound on the size of any clique
        # that can be formed from P.
        # The color of a vertex is the smallest integer c such that it is not adjacent
        # to any vertex of color < c? No, standard greedy coloring for max clique bound:
        # We assign colors such that adjacent vertices have different colors.
        # Then any clique can contain at most one vertex of each color.
        # So the number of colors is an upper bound on the clique size in P.
        
        # To make the bound tight, we should color in an order that allows
        # small colors for high-degree vertices? Or reverse?
        # Usually, we sort by degree in P (descending) and color greedily.
        
        verts.sort(key=lambda v: (P & adj[v]).bit_count(), reverse=True)
        
        colors = {}
        color_classes = [] # list of bitmasks, color_classes[c] has all vertices with color c
        
        for v in verts:
            c = 0
            # Find smallest color c such that v is not adjacent to any vertex with color c
            # Wait, the condition is: v must not be adjacent to any vertex *already colored* with color c.
            # Since we are coloring greedily, we check against the current color classes.
            while c < len(color_classes) and (adj[v] & color_classes[c]):
                c += 1
            if c == len(color_classes):
                color_classes.append(0)
            color_classes[c] |= (1 << v)
            colors[v] = c
            
        max_color = len(color_classes) - 1
        
        # Search in reverse order of colors (from highest color to lowest)
        # Why? Because vertices with higher colors are likely to be in smaller cliques?
        # Actually, the standard trick is to iterate vertices in the order they were colored,
        # but process them in reverse color order to find large cliques faster?
        # Or simply: The bound for a vertex v with color c is len(C) + c + 1.
        # If len(C) + c + 1 <= best_len, we can skip v.
        
        # We iterate through vertices in the order of decreasing color?
        # Let's iterate through the color classes from max_color down to 0.
        # For each class, iterate through vertices.
        
        for c in range(max_color, -1, -1):
            # Check bound for this color class
            # Any clique extending C using vertices from color c or lower has size at most len(C) + c + 1
            # (since colors are 0..max_color, and we've processed higher colors? No, we are processing c now)
            # Actually, the bound is: if we pick a vertex of color c, the max additional size is c+1?
            # No, the greedy coloring ensures that the subgraph induced by vertices of color <= c
            # has a clique number at most c+1? No, that's not right.
            
            # Correct logic:
            # The greedy coloring assigns color c to v. This means v is not adjacent to any vertex with color < c?
            # No, the condition `while c < len(color_classes) and (adj[v] & color_classes[c]): c += 1`
            # means v is adjacent to some vertex in color 0, some in color 1, ..., some in color c-1.
            # So v is NOT adjacent to any vertex in color c? No, it means it *was* adjacent to vertices in 0..c-1,
            # so it needed color c.
            # This implies that the set of vertices with color <= c contains a clique of size at most c+1?
            # Actually, a known property: If vertices are colored greedily in a specific order,
            # the number of colors is an upper bound on the clique size.
            # More specifically, if we process vertices in reverse color order,
            # we can prune.
            
            if len(C) + c + 1 <= best_len:
                # Since colors are decreasing, all subsequent vertices have color <= c.
                # So any clique found will have size <= len(C) + c + 1 <= best_len.
                # We can break out of the loop entirely.
                break
                
            class_verts = []
            temp_C = color_classes[c]
            while temp_C:
                lsb = temp_C & -temp_C
                v = lsb.bit_length() - 1
                class_verts.append(v)
                temp_C ^= lsb
                
            # Process vertices in this color class
            # We should process them in the order they were added? Or reverse?
            # Usually, we iterate and update P.
            for v in class_verts:
                # Check if v is still in P (it might have been removed by previous iterations? No, P is local to this expand call, 
                # but we are modifying P? No, we are passing P by value (int). 
                # Wait, in the loop `for v in class_verts`, we are not modifying P yet.
                # We need to modify P to exclude v for subsequent branches?
                # Yes, standard BK: 
                # for v in P:
                #    expand(C + v, P & N(v))
                #    P = P \ {v}
                
                new_C = C + [v]
                new_P = P & adj[v]
                
                expand(new_C, new_P)
                
                # Remove v from P for the next iteration of the loop
                P &= ~(1 << v)
                
                # Re-evaluate bound: if len(C) + P.bit_count() <= best_len, we can return
                if len(C) + P.bit_count() <= best_len:
                    return

    expand([], init_mask)
    return best
