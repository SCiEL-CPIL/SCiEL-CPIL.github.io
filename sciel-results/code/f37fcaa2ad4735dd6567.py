import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin
    
    # 1. Construct Grötzsch graph on vertices 0..10
    # Cycle 0-1-2-3-4-0
    # New vertices 5..9
    # Vertex 10 connected to 5..9
    # Edges:
    # (0,1), (1,2), (2,3), (3,4), (4,0)
    # (0,6), (1,5), (1,7), (2,6), (2,8), (3,7), (3,9), (4,8), (4,9) -- wait, let's use the standard construction
    # Mycielski of C5.
    # Original vertices 0..4.
    # New vertices 5..9.
    # Vertex 10.
    # Edges in C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Edges from new to old:
    # 5 connected to 1, 4 (neighbors of 0)
    # 6 connected to 0, 2 (neighbors of 1)
    # 7 connected to 1, 3 (neighbors of 2)
    # 8 connected to 2, 4 (neighbors of 3)
    # 9 connected to 3, 0 (neighbors of 4)
    # Edges from 10 to 5,6,7,8,9
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 1], [5, 4],
        [6, 0], [6, 2],
        [7, 1], [7, 3],
        [8, 2], [8, 4],
        [9, 3], [9, 0],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    # Convert to adjacency list/mask for fast operations
    # Vertices 0..15
    adj = [0] * 16
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # 2. Maximize edges in the subgraph on 0..10
    # Try to add all missing edges if they don't create a triangle
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for u in range(11):
            for v in range(u + 1, 11):
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates a triangle
                    # Common neighbors
                    if (adj[u] & adj[v]) == 0:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
                        edges.append([u, v])
                        
    # 3. Find all independent sets in the graph on 0..10
    # An independent set S is a subset of {0..10} such that no two vertices in S are adjacent.
    independent_sets = []
    for mask in range(1 << 11):
        is_independent = True
        # Check if mask is independent
        # For each vertex u in mask, adj[u] & mask should be 0?
        # No, adj[u] contains neighbors. If any neighbor is in mask, not independent.
        # So (adj[u] & mask) == 0 for all u in mask?
        # Actually, if u in mask, and v in mask, and u~v, then v is in adj[u].
        # So we just need to check if any edge exists within mask.
        # Faster: 
        # If mask is independent, then for all u in mask, (adj[u] & mask) == 0 is NOT correct because adj[u] doesn't include u.
        # Wait, adj[u] & mask checks if any neighbor of u is in mask.
        # If u is in mask, and v is in mask and u~v, then v is in adj[u] and v is in mask.
        # So (adj[u] & mask) will be non-zero.
        # So yes, condition is: for all u in mask, (adj[u] & mask) == 0.
        # But we can just check: 
        # Let m = mask.
        # If m is independent, then no edge (u,v) with u,v in m.
        # We can precompute edge masks?
        # Simpler:
        # Iterate u in 0..10. If u in mask, check if adj[u] & mask == 0.
        # This is O(11) per mask. 2048 * 11 is fine.
        
        temp_mask = mask
        while temp_mask:
            u = (temp_mask & -temp_mask).bit_length() - 1
            if adj[u] & mask:
                is_independent = False
                break
            temp_mask &= temp_mask - 1
            
        if is_independent:
            independent_sets.append(mask)
            
    # Sort independent sets by size (popcount) descending
    independent_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # 4. Iterate over all possible graphs G_new on vertices 11..15
    # There are 5 vertices, 10 possible edges.
    # We need G_new to be triangle-free.
    # For each valid G_new, we need to assign independent sets S_11..S_15 from independent_sets
    # to maximize sum(|S_i|) subject to S_u & S_v == 0 for all edges (u,v) in G_new.
    
    best_total_edges = len(edges)
    best_config = None
    
    # Precompute edges for G_new candidates
    # Vertices 11..15 correspond to indices 0..4 in local G_new
    # We iterate over all 2^10 = 1024 masks for edges in G_new
    
    for g_mask in range(1 << 10):
        if time.time() > deadline:
            break
            
        # Check if G_new is triangle-free
        # Edges are bits 0..9
        # (0,1)=bit0, (0,2)=bit1, (0,3)=bit2, (0,4)=bit3
        # (1,2)=bit4, (1,3)=bit5, (1,4)=bit6
        # (2,3)=bit7, (2,4)=bit8
        # (3,4)=bit9
        
        # Triangles in K5:
        # (0,1,2): bits 0, 1, 4
        # (0,1,3): bits 0, 2, 5
        # (0,1,4): bits 0, 3, 6
        # (0,2,3): bits 1, 7, 4? No. (0,2) is 1, (2,3) is 7, (0,3) is 2.
        # Let's list triangles by vertex sets:
        # {0,1,2}: (0,1), (1,2), (0,2) -> bits 0, 4, 1
        # {0,1,3}: (0,1), (1,3), (0,3) -> bits 0, 5, 2
        # {0,1,4}: (0,1), (1,4), (0,4) -> bits 0, 6, 3
        # {0,2,3}: (0,2), (2,3), (0,3) -> bits 1, 7, 2
        # {0,2,4}: (0,2), (2,4), (0,4) -> bits 1, 8, 3
        # {0,3,4}: (0,3), (3,4), (0,4) -> bits 2, 9, 3
        # {1,2,3}: (1,2), (2,3), (1,3) -> bits 4, 7, 5
        # {1,2,4}: (1,2), (2,4), (1,4) -> bits 4, 8, 6
        # {1,3,4}: (1,3), (3,4), (1,4) -> bits 5, 9, 6
        # {2,3,4}: (2,3), (3,4), (2,4) -> bits 7, 9, 8
        
        triangles = [
            (0, 1, 4), (0, 2, 5), (0, 3, 6),
            (1, 2, 7), (1, 3, 8), (2, 3, 9),
            (4, 5, 7), (4, 6, 8), (5, 6, 9), (7, 8, 9)
        ]
        
        is_tf = True
        for t in triangles:
            if (g_mask & (1 << t[0])) and (g_mask & (1 << t[1])) and (g_mask & (1 << t[2])):
                is_tf = False
                break
                
        if not is_tf:
            continue
            
        e_new = bin(g_mask).count('1')
        
        # Solve assignment problem for this G_new
        # Vertices 0..4 in G_new correspond to 11..15
        # We need to assign independent sets S_0..S_4
        # Constraints: S_u & S_v == 0 for edges (u,v) in G_new
        
        # Backtracking
        # Order vertices by degree in G_new (highest first)
        degrees = [0] * 5
        for i in range(5):
            for j in range(i + 1, 5):
                bit = 0
                # Map (i,j) to bit index
                # 0,1->0; 0,2->1; 0,3->2; 0,4->3
                # 1,2->4; 1,3->5; 1,4->6
                # 2,3->7; 2,4->8
                # 3,4->9
                if i == 0 and j == 1: bit = 0
                elif i == 0 and j == 2: bit = 1
                elif i == 0 and j == 3: bit = 2
                elif i == 0 and j == 4: bit = 3
                elif i == 1 and j == 2: bit = 4
                elif i == 1 and j == 3: bit = 5
                elif i == 1 and j == 4: bit = 6
                elif i == 2 and j == 3: bit = 7
                elif i == 2 and j == 4: bit = 8
                elif i == 3 and j == 4: bit = 9
                
                if g_mask & (1 << bit):
                    degrees[i] += 1
                    degrees[j] += 1
                    
        order = sorted(range(5), key=lambda x: degrees[x], reverse=True)
        
        best_sum = -1
        best_assignment = [0] * 5
        
        def backtrack(idx, current_sum, assignment, used_masks):
            nonlocal best_sum, best_assignment
            if time.time() > deadline:
                return
                
            if idx == 5:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_assignment = assignment[:]
                return
                
            # Pruning: if current_sum + max_possible_remaining <= best_sum, return
            # max_possible_remaining: sum of sizes of largest independent sets
            # This is a weak bound, but okay.
            
            u = order[idx]
            
            # Try independent sets
            for s in independent_sets:
                # Check constraints with already assigned neighbors
                valid = True
                for v in range(5):
                    if v == u: continue
                    if assignment[v] is not None:
                        # Check if u and v are connected in G_new
                        # Need bit index for (u,v)
                        i, j = sorted([u, v])
                        bit = 0
                        if i == 0 and j == 1: bit = 0
                        elif i == 0 and j == 2: bit = 1
                        elif i == 0 and j == 3: bit = 2
                        elif i == 0 and j == 4: bit = 3
                        elif i == 1 and j == 2: bit = 4
                        elif i == 1 and j == 3: bit = 5
                        elif i == 1 and j == 4: bit = 6
                        elif i == 2 and j == 3: bit = 7
                        elif i == 2 and j == 4: bit = 8
                        elif i == 3 and j == 4: bit = 9
                        
                        if g_mask & (1 << bit):
                            if s & assignment[v]:
                                valid = False
                                break
                                
                if valid:
                    assignment[u] = s
                    backtrack(idx + 1, current_sum + bin(s).count('1'), assignment, used_masks)
                    assignment[u] = None
                    
                    # Early exit if we found a perfect solution?
                    # Max size is 4. 5*4=20.
                    # If best_sum == 20, we can stop?
                    # But we are in a loop over g_mask.
                    # Inside backtrack, we can stop if we hit the theoretical max.
                    if best_sum == 20:
                        return
                        
        backtrack(0, 0, [None]*5, None)
        
        if best_sum > -1:
            total = len(edges) + e_new + best_sum
            if total > best_total_edges:
                best_total_edges = total
                best_config = (g_mask, best_assignment)
                
    # 5. Construct final edge list
    if best_config is None:
        # Fallback: just return the base graph with isolated new vertices
        # This is valid (Grötzsch is not 3-colorable)
        return edges
        
    g_mask, assignment = best_config
    
    # Add edges from new vertices to base
    for i in range(5):
        v_new = 11 + i
        s = assignment[i]
        # s is a mask of vertices in 0..10
        while s:
            u = (s & -s).bit_length() - 1
            edges.append([v_new, u])
            s &= s - 1
            
    # Add edges within new vertices
    for i in range(5):
        for j in range(i + 1, 5):
            # Map to bit
            if i == 0 and j == 1: bit = 0
            elif i == 0 and j == 2: bit = 1
            elif i == 0 and j == 3: bit = 2
            elif i == 0 and j == 4: bit = 3
            elif i == 1 and j == 2: bit = 4
            elif i == 1 and j == 3: bit = 5
            elif i == 1 and j == 4: bit = 6
            elif i == 2 and j == 3: bit = 7
            elif i == 2 and j == 4: bit = 8
            elif i == 3 and j == 4: bit = 9
            else: bit = -1
            
            if g_mask & (1 << bit):
                edges.append([11 + i, 11 + j])
                
    return edges
