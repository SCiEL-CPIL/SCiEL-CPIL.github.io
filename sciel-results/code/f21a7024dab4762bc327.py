import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    N = 16

    # 1. Define Grötzsch Graph (Mycielski of C5)
    # Vertices 0..10
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [5, 0], [5, 2],                          # Copy 5 of 0
        [6, 1], [6, 3],                          # Copy 6 of 1
        [7, 2], [7, 4],                          # Copy 7 of 2
        [8, 3], [8, 0],                          # Copy 8 of 3
        [9, 4], [9, 1],                          # Copy 9 of 4
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9] # Top 10
    ]
    
    # Initialize adjacency bitmasks
    best_adj = [0] * N
    best_edge_count = 0
    
    # Base graph edges
    base_adj = [0] * N
    for u, v in groetzsch_edges:
        base_adj[u] |= (1 << v)
        base_adj[v] |= (1 << u)
    base_edge_count = len(groetzsch_edges)
    
    # Pre-compute all possible edges not in base graph
    # u < v
    all_possible_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if not (base_adj[u] & (1 << v)):
                all_possible_edges.append((u, v))
                
    # Helper to check 3-colorability using bitmasks
    # Returns True if NOT 3-colorable
    def is_not_3_colorable(adj):
        color = [-1] * N
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(N)]
        order = sorted(range(N), key=lambda x: -degrees[x])
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Available colors
            used = 0
            mask = adj[v]
            while mask:
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used |= (1 << color[u])
                mask ^= lsb
            
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack(0)

    # Helper to find Maximum Independent Set (MIS) in a graph represented by adj bitmasks
    # restricted to a subset of vertices (given as a bitmask)
    def find_mis(adj, vertices_mask):
        # vertices_mask is a bitmask of vertices to consider
        best_set = 0
        best_size = 0
        
        # Order vertices by degree in the induced subgraph (descending)
        # This helps pruning
        verts = []
        mask = vertices_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            # Count neighbors within the induced subgraph
            deg = bin(adj[v] & vertices_mask).count('1')
            verts.append((deg, v))
            mask ^= lsb
            
        verts.sort(reverse=True)
        ordered_verts = [v for _, v in verts]
        
        def backtrack(idx, current_mask, current_set):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
                
            if idx == len(ordered_verts):
                if len(bin(current_set).count('1')) > best_size:
                    best_size = len(bin(current_set).count('1'))
                    best_set = current_set
                return
                
            # Upper bound: current size + remaining vertices
            if len(bin(current_set).count('1')) + (len(ordered_verts) - idx) <= best_size:
                return
                
            v = ordered_verts[idx]
            
            # Option 1: Include v
            if current_mask & (1 << v):
                # Remove v and its neighbors from consideration
                new_mask = current_mask & ~(adj[v] | (1 << v))
                backtrack(idx + 1, new_mask, current_set | (1 << v))
                
            # Option 2: Exclude v
            backtrack(idx + 1, current_mask, current_set)
            
        backtrack(0, vertices_mask, 0)
        
        # Convert best_set bitmask to list of vertices
        result = []
        mask = best_set
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            result.append(v)
            mask ^= lsb
        return result

    # Phase 1: Greedy addition with randomization (from current program)
    # We will run this for a portion of the time
    greedy_deadline = start_time + 30.0
    
    while time.time() < greedy_deadline:
        # Reset to base
        adj = base_adj[:]
        edge_count = base_edge_count
        
        # Shuffle the candidate edges
        candidates = all_possible_edges[:]
        random.shuffle(candidates)
        
        # Greedy addition
        for u, v in candidates:
            # Check if adding (u,v) creates a triangle
            if (adj[u] & adj[v]) == 0:
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Check if this graph is not 3-colorable
        if is_not_3_colorable(adj):
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_adj = adj[:]

    # Phase 2: Systematic addition of vertices 11-15 using MIS (from inspiration 1)
    # Start from the best graph found in Phase 1, or if that failed, use base
    if best_edge_count == 0:
        adj = base_adj[:]
    else:
        adj = best_adj[:]
        
    # Vertices 11-15 are initially isolated in the base graph, but might have edges in Phase 1 results.
    # We need to be careful. The inspiration approach builds up from 11 to 15.
    # Let's try to improve the current best graph by potentially adding edges involving 11-15 more intelligently?
    # Actually, the inspiration method constructs the graph layer by layer.
    # Let's try a hybrid: Use the greedy result as a starting point, then try to add more edges
    # or use the MIS method to construct a different candidate and compare.
    
    # Let's implement the MIS-based construction separately and compare.
    mis_adj = base_adj[:]
    
    # Vertices 11-15
    for v in range(11, 16):
        if time.time() > deadline:
            break
            
        # Find MIS in vertices 0..v-1
        # Mask for 0..v-1
        mask = (1 << v) - 1
        mis = find_mis(mis_adj, mask)
        
        # Connect v to all vertices in MIS
        for u in mis:
            # Check triangle-free
            # Adding (v, u). Triangle if N(v) & N(u) != 0.
            # N(v) is currently empty (since v is new and we process sequentially).
            # Wait, if we add edges to v, N(v) grows.
            # But we are adding all edges to v at once.
            # So for each u in MIS, we add (v, u).
            # Since MIS is an independent set, no two u's in MIS are adjacent.
            # So adding (v, u1) and (v, u2) does not create a triangle between u1-v-u2 because u1-u2 is not an edge.
            # Also, v is not connected to any other vertex yet, so no other triangles.
            # So it is safe to add all edges from v to MIS.
            mis_adj[v] |= (1 << u)
            mis_adj[u] |= (1 << v)
            
    # Check if MIS graph is valid and better
    mis_edge_count = sum(bin(mis_adj[i]).count('1') for i in range(N)) // 2
    
    # Verify triangle-free for MIS graph (should be, but just in case)
    is_valid = True
    for i in range(N):
        for j in range(i+1, N):
            if (mis_adj[i] & (1 << j)) and (mis_adj[j] & (1 << i)):
                if (mis_adj[i] & mis_adj[j]):
                    is_valid = False
                    break
        if not is_valid:
            break
            
    if is_valid and is_not_3_colorable(mis_adj) and mis_edge_count > best_edge_count:
        best_edge_count = mis_edge_count
        best_adj = mis_adj[:]

    # Phase 3: Local search / Polish
    # Try to add edges to the best graph found so far
    # Use a randomized local search or greedy fill on the remaining edges
    
    # Identify edges that can still be added to best_adj
    remaining_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if not (best_adj[u] & (1 << v)):
                if (best_adj[u] & best_adj[v]) == 0:
                    remaining_edges.append((u, v))
                    
    if remaining_edges:
        # Randomized greedy fill on the best graph
        polish_deadline = deadline - 2.0
        
        current_best_adj = best_adj[:]
        current_best_count = best_edge_count
        
        while time.time() < polish_deadline:
            adj = current_best_adj[:]
            edge_count = current_best_count
            
            candidates = remaining_edges[:]
            random.shuffle(candidates)
            
            # Try to add as many as possible
            for u, v in candidates:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
            
            if is_not_3_colorable(adj) and edge_count > current_best_count:
                current_best_count = edge_count
                current_best_adj = adj[:]
                
                # Update global best
                if current_best_count > best_edge_count:
                    best_edge_count = current_best_count
                    best_adj = current_best_adj[:]
                    
                # Recompute remaining edges for this new graph
                remaining_edges = []
                for u in range(N):
                    for v in range(u + 1, N):
                        if not (current_best_adj[u] & (1 << v)):
                            if (current_best_adj[u] & current_best_adj[v]) == 0:
                                remaining_edges.append((u, v))
                                
                if not remaining_edges:
                    break

    # Convert best_adj to list of edges
    result_edges = []
    for u in range(N):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                result_edges.append([u, v])
            mask ^= lsb
            
    return result_edges
