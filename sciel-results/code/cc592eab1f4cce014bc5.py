import random
import time

def construct():
    n_base = 29
    q = n_base
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    base_colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                base_colors.append(0)
            else:
                base_colors.append(1)
                
    if not is_valid(n_base, base_colors):
        return get_paley(17)
        
    # Local search to extend to n=30
    # New vertex 29. Edges (i, 29) for i=0..28.
    # We represent the state as a list of 29 colors.
    
    start_time = time.time()
    time_limit = 110.0
    
    # Initialize with random colors
    new_edges = [random.randint(0, 1) for _ in range(n_base)]
    
    best_score = -1
    best_solution = None
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Build full colors list for n=30
        full_colors = build_full_colors(base_colors, new_edges)
        
        # Check validity
        if is_valid(30, full_colors):
            return {'n': 30, 'colors': full_colors}
            
        # Find a violation (K2,2,2)
        # To save time, we only check for K2,2,2 involving the new vertex?
        # No, flipping an edge might fix a violation not involving the new vertex?
        # No, the base graph is valid. Only violations involving the new vertex can appear.
        # So we only need to check for K2,2,2 involving vertex 29.
        
        violation = find_k222_involving_vertex(30, full_colors, 29)
        
        if violation:
            # Flip one of the edges in the violation that involves vertex 29
            # The violation is a set of 6 vertices.
            # We need to flip an edge (u, 29) where u is in the violation.
            # Or flip an edge between two existing vertices? No, those are fixed.
            # So we must flip an edge (u, 29).
            
            # Identify which edges in the violation involve 29
            # The violation returns the 6 vertices.
            # We pick a random vertex u from the violation (u != 29) and flip edge (u, 29).
            # Actually, we should flip an edge that is part of the K2,2,2 structure.
            # All edges between parts are in the K2,2,2.
            # Edges involving 29 are (29, v) for v in the other parts.
            
            # Let's just flip a random edge (u, 29) where u is in the violation set.
            candidates = [v for v in violation if v != 29]
            if candidates:
                u = random.choice(candidates)
                new_edges[u] = 1 - new_edges[u]
        else:
            # No violation involving 29? Then it should be valid.
            # If is_valid says false, there might be a bug or a violation not involving 29 (impossible).
            # Break to avoid infinite loop.
            break

    # If no solution found, return base
    return {'n': n_base, 'colors': base_colors}

def build_full_colors(base_colors, new_edges):
    # base_colors is for n=29
    # new_edges is for (i, 29)
    # Full colors for n=30
    full_colors = []
    idx_base = 0
    for i in range(30):
        for j in range(i + 1, 30):
            if i < 29 and j < 29:
                full_colors.append(base_colors[idx_base])
                idx_base += 1
            else:
                # j must be 29, i < 29
                full_colors.append(new_edges[i])
    return full_colors

def find_k222_involving_vertex(n, colors, v):
    # Check for K2,2,2 involving vertex v
    # Build adjacency for colors 0 and 1
    adj0 = [set() for _ in range(n)]
    adj1 = [set() for _ in range(n)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i].add(j)
                adj0[j].add(i)
            else:
                adj1[i].add(j)
                adj1[j].add(i)
            idx += 1
            
    # Check color 0
    res = find_k222_in_adj(n, adj0, v)
    if res:
        return res
        
    # Check color 1
    res = find_k222_in_adj(n, adj1, v)
    if res:
        return res
        
    return None

def find_k222_in_adj(n, adj, v):
    # Find K2,2,2 involving v in graph adj
    # v is in one of the parts.
    # Let v be in part A. A = {v, a}.
    # a is NOT connected to v in adj.
    # So a must be a non-neighbor of v.
    
    non_neighbors = [u for u in range(n) if u != v and u not in adj[v]]
    
    for a in non_neighbors:
        # A = {v, a}
        # B = {b1, b2}, C = {c1, c2}
        # b1, b2, c1, c2 must be neighbors of v and a.
        # So they must be in adj[v] & adj[a].
        
        common = adj[v] & adj[a]
        
        # We need to pick b1, b2 from common such that they are NOT connected.
        # And c1, c2 from common \ {b1, b2} such that they are NOT connected,
        # and all cross edges between {b1, b2} and {c1, c2} exist.
        
        # Iterate over pairs (b1, b2) in common
        common_list = list(common)
        for i in range(len(common_list)):
            b1 = common_list[i]
            for j in range(i + 1, len(common_list)):
                b2 = common_list[j]
                if b1 in adj[b2]:
                    continue # Must be non-adjacent
                
                # Find candidates for C
                candidates = []
                for k in range(len(common_list)):
                    c = common_list[k]
                    if c == b1 or c == b2:
                        continue
                    if c in adj[b1] and c in adj[b2]:
                        candidates.append(c)
                        
                if len(candidates) >= 2:
                    # Check for pair (c1, c2) in candidates that are non-adjacent
                    for x in range(len(candidates)):
                        c1 = candidates[x]
                        for y in range(x + 1, len(candidates)):
                            c2 = candidates[y]
                            if c1 not in adj[c2]:
                                return [v, a, b1, b2, c1, c2]
    return None

def is_valid(n, colors):
    # Full validity check
    adj0 = [set() for _ in range(n)]
    adj1 = [set() for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i].add(j)
                adj0[j].add(i)
            else:
                adj1[i].add(j)
                adj1[j].add(i)
            idx += 1
            
    if has_k222(n, adj0):
        return False
    if has_k222(n, adj1):
        return False
    return True

def has_k222(n, adj):
    # General check
    for u in range(n):
        for v in range(u + 1, n):
            if u in adj[v]:
                continue # u, v must be non-adjacent in the part?
                # Wait, in K2,2,2, vertices in the same part are non-adjacent.
                # So if u, v are adjacent, they cannot be in the same part.
                # But they could be in different parts.
                # My previous logic assumed u, v are in the same part.
                # This function needs to check all possibilities.
                # This is slow.
                # For the purpose of this task, I will rely on the specific
                # "involving vertex" check for the search, and assume the base is valid.
                # The full is_valid is only used for the final check.
                pass
                
    # Simplified check for final validation
    # Since n=30 is small, we can do a more thorough check here if needed.
    # But the "involving vertex" check is sufficient for the search loop.
    # For the final return, we must be sure.
    
    # Re-implementing a faster general check is risky.
    # I will use the existing slow one but only call it once at the end.
    
    # Actually, the existing has_k222 in the prompt is buggy/slow.
    # I will write a correct one for n=30.
    
    # A K2,2,2 is a 6-vertex subgraph.
    # There are C(30, 6) ~ 593775 ways to pick 6 vertices.
    # For each set, check if it induces a K2,2,2 in color 0 or 1.
    # This is feasible.
    
    from itertools import combinations
    
    for comb in combinations(range(n), 6):
        # Check if this set of 6 vertices forms a K2,2,2 in color 0
        if is_k222_subset(comb, adj, 0):
            return True
        if is_k222_subset(comb, adj, 1):
            return True
            
    return False

def is_k222_subset(vertices, adj, color):
    # vertices is a tuple of 6 ints
    # Check if they can be partitioned into 3 pairs such that
    # edges within pairs are 1-color, edges between pairs are color.
    
    # Try all 15 partitions of 6 items into 3 pairs
    # Partition 1: (0,1), (2,3), (4,5)
    # We can just check if the induced subgraph on these 6 vertices
    # is isomorphic to K2,2,2.
    
    # K2,2,2 has 12 edges between parts, 0 edges within parts.
    # So the induced subgraph must have exactly 12 edges of the given color.
    # And the 6 non-edges must be the 3 pairs.
    
    # Let's check the degree sequence or edge count.
    # In K2,2,2, each vertex has degree 4 (connected to 4 vertices in other parts).
    # So all 6 vertices must have degree 4 in the induced subgraph of color.
    
    induced_deg = [0] * 6
    for i in range(6):
        for j in range(i + 1, 6):
            u = vertices[i]
            v = vertices[j]
            if u in adj[v]: # Adjacency in the graph passed to has_k222
                induced_deg[i] += 1
                induced_deg[j] += 1
                
    if all(d == 4 for d in induced_deg):
        return True
        
    return False

def get_paley(n):
    q = n
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in squares:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
