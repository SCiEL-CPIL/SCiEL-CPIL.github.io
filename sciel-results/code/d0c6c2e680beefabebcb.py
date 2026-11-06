import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Precompute all possible edges and index them
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)  # 190 edges
    
    # edge_index: (u,v) -> index
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Build conflict graph: two edges conflict if they cannot both be in K2,3-free graph
    # Conflict conditions:
    # 1. Edges sharing a vertex: (u,v) and (u,w) - no direct conflict yet
    # 2. (u,v) and (u,w): if they have 2 common neighbors, adding both creates K2,3
    # 3. (u,v) and (w,x): if u,w have 2 common neighbors including v,x... 
    # Actually, let's think more carefully.
    # K2,3 free means: no two vertices share 3+ common neighbors.
    # Two edges (a,b) and (c,d) conflict if adding both creates a K2,3.
    # This happens if:
    # - They share a vertex, say (a,b) and (a,c): then b and c must not have 2 common neighbors
    #   (since a is already a common neighbor, plus any other)
    # - They are disjoint: (a,b) and (c,d): need to check various conditions
    
    # Simpler approach: use greedy + local search with fast validation
    # But also try MCIS for the conflict structure
    
    # For MCIS: node i represents edge i. Two nodes conflict if both edges can't coexist.
    # Build conflict adjacency as bitmasks
    conflict = [0] * m
    
    # Precompute for each edge (u,v), the set of other edges that conflict with it
    for i in range(m):
        u1, v1 = all_edges[i]
        for j in range(i + 1, m):
            u2, v2 = all_edges[j]
            # Check if edges (u1,v1) and (u2,v2) can coexist in K2,3-free graph
            # They conflict if there exist vertices x,y such that {x,y} have 3 common neighbors
            # when both edges are present.
            
            # Case 1: share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                # Shared vertex case
                # (u1,v1) and (u2,v2) share a vertex
                # The non-shared endpoints must not have 2 common neighbors
                if u1 == u2:
                    # (u1,v1) and (u1,v2): v1 and v2 must not have 2 common neighbors
                    # Common neighbors of v1 and v2 (excluding u1 which is connected to both)
                    # In the full graph, v1 and v2 would share u1 as a common neighbor
                    # They'd have 3 common neighbors if they share 2 more
                    # But we're checking pairwise conflict, so:
                    # If v1 and v2 have 2 common neighbors (not counting u1), then
                    # adding both edges gives v1,v2 three common neighbors (u1 + 2 others)
                    # However, those "others" need to actually be in the graph.
                    # For pairwise conflict in MCIS, we need a simpler criterion.
                    # 
                    # Actually for MCIS on conflict graph, we need: edges i and j conflict
                    # if NO valid K2,3-free graph can contain both.
                    # This is hard to determine a priori without knowing the full graph.
                    # 
                    # Alternative: use a different approach.
                    # Let's stick with greedy + local search but make it much faster.
                    pass
                continue
            
            # Case 2: disjoint edges (u1,v1) and (u2,v2)
            # These could create K2,3 if:
            # - u1 and u2 have v1,v2 as common neighbors (2 already, need 1 more)
            # - u1 and v2 have u2,v1 as common neighbors
            # - v1 and v2 have u1,u2 as common neighbors
            # - v1 and u2 have v2,u1 as common neighbors
            # In any case, if two pairs of vertices each have 2 common neighbors that are
            # the endpoints of the other edge, then adding both creates K2,3.
            
            # Actually, the precise condition is complex. Let me use a different strategy:
            # Fast greedy with bitmasks + aggressive local search.
            pass
    
    # Given the complexity of exact conflict detection, let's use an efficient
    # greedy + local search approach with very fast edge validation using bitmasks.
    
    best_edges = []
    best_count = 0
    
    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj. Returns True if valid."""
        # Condition 1: u and v must have at most 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # Condition 2: For each neighbor b of v (in current graph), 
        # u and b must have at most 1 common neighbor
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        # Condition 3: For each neighbor a of u (in current graph),
        # v and a must have at most 1 common neighbor
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        return True
    
    def try_add_edge(adj, current_edges, u, v):
        """Try to add edge (u,v). Returns True if added."""
        if validate_and_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.append((u, v))
            return True
        return False
    
    def remove_edge(adj, current_edges, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        current_edges.remove((u, v))
    
    def local_search(adj, current_edges, max_iter=500):
        """Try to improve by adding edges, then swapping."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # Try adding any missing edge
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                # Try to add a new edge
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u] & (1 << w):
                        continue  # already connected
                    if validate_and_add(adj, u, w):
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        current_edges.append((u, w))
                        improved = True
                        break
                if improved:
                    break
            
            # Try edge swaps: remove one, add two
            if not improved and len(current_edges) > 0:
                for i in range(len(current_edges)):
                    u, v = current_edges[i]
                    old_adj = adj[:]
                    old_edges = current_edges[:]
                    remove_edge(adj, current_edges, u, v)
                    
                    added = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[u] & (1 << w) or adj[v] & (1 << w):
                            continue
                        if validate_and_add(adj, u, w):
                            adj[u] |= (1 << w)
                            adj[w] |= (1 << u)
                            current_edges.append((u, w))
                            added += 1
                        if added >= 2:
                            break
                        if validate_and_add(adj, v, w):
                            adj[v] |= (1 << w)
                            adj[w] |= (1 << v)
                            current_edges.append((v, w))
                            added += 1
                        if added >= 2:
                            break
                    
                    if added < 2:
                        # Revert
                        adj[:] = old_adj
                        current_edges[:] = old_edges
                    else:
                        improved = True
                        break
        
        return current_edges
    
    # Main loop: multiple restarts with different orderings
    while time.time() < deadline:
        # Greedy construction with random ordering
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        # Local search to improve
        current_edges = local_search(adj, current_edges, max_iter=200)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # Try more aggressive local search on the best found
            if time.time() < deadline:
                adj2 = [0] * n
                for u, v in best_edges:
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
                best_edges = local_search(adj2, best_edges, max_iter=500)
                if len(best_edges) > best_count:
                    best_count = len(best_edges)
    
    # Convert list of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
