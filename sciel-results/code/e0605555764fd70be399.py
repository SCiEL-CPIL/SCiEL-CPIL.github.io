import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Keep 10s margin for 120s limit
    
    n = 16
    # Precompute all 8-vertex subsets
    subsets = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        for o in range(m+1, n):
                            for p in range(o+1, n):
                                for q in range(p+1, n):
                                    subsets.append((i, j, k, l, m, o, p, q))
                                    
    # Q3 edges (0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7)
    q3_edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    def contains_q3(adj, subset):
        # adj: list of bitmasks
        # subset: tuple of 8 vertices
        # Check if induced subgraph on subset contains a Q3
        
        # 1. Check degrees
        # Map subset vertices to 0..7
        # For each vertex in subset, count neighbors within subset
        # If any count < 3, return False
        
        # Optimization: Only check if the number of edges is >= 12
        # But we need to check if a Q3 *subgraph* exists.
        
        # Fast check:
        # If the induced subgraph is not bipartite, it might still contain a Q3.
        # But Q3 is bipartite.
        
        # Let's just check if there is a 3-regular spanning subgraph.
        # This is equivalent to: Can we select 12 edges such that every vertex has degree 3?
        
        # Given the small size, we can try to find such a subgraph.
        # If the number of edges in induced subgraph < 12, return False.
        
        edges_in_subset = 0
        for u in subset:
            # Count neighbors in subset
            mask = 0
            for v in subset:
                if u != v and (adj[u] >> v) & 1:
                    mask |= (1 << v)
            edges_in_subset += bin(mask).count('1')
        edges_in_subset //= 2
        
        if edges_in_subset < 12:
            return False
            
        # Check if min degree >= 3
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and (adj[u] >> v) & 1:
                    deg += 1
            if deg < 3:
                return False
                
        # If we get here, it's a candidate.
        # Check if it contains a Q3.
        # Since min degree >= 3 and edges >= 12, it's very likely.
        # We can check if it is isomorphic to Q3 or has a Q3 subgraph.
        # For simplicity and speed, we assume that if min degree >= 3 and edges >= 12,
        # and the graph is "dense enough", it contains a Q3.
        # Actually, the only 3-regular graph on 8 vertices is Q3 (and its isomorphs).
        # So if we can find a 3-regular spanning subgraph, it is Q3.
        
        # How to check for 3-regular spanning subgraph?
        # We can try to remove edges to make it 3-regular.
        # If edges == 12, check if it is 3-regular.
        if edges_in_subset == 12:
            # Check if 3-regular
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and (adj[u] >> v) & 1:
                        deg += 1
                if deg != 3:
                    return False
            return True
            
        # If edges > 12, we need to check if a 3-regular spanning subgraph exists.
        # This is always true for 8-vertex graphs with min degree >= 3?
        # No.
        # But we can just return True to be safe?
        # If we return True, we might remove edges unnecessarily.
        # If we return False, we might miss a Q3.
        
        # Let's do a quick check:
        # If the graph is bipartite, and min degree >= 3, it contains a Q3?
        # Not necessarily.
        
        # Given the time constraint, I will assume that if min degree >= 3 and edges >= 12,
        # it contains a Q3. This is a conservative approximation.
        # Wait, if it doesn't contain a Q3, we are safe.
        # If it does, we need to remove an edge.
        
        # To be safe, I will return True if edges >= 12 and min degree >= 3.
        # This might lead to over-removal, but it ensures validity.
        return True

    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.6:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_edges = -1
    best_adj = adj[:]
    
    # Find initial Q3s and remove edges
    # We will run a loop
    while time.time() - start_time < time_limit:
        # Check if current graph is Q3-free
        is_free = True
        found_q3_subset = None
        found_q3_edges = []
        
        for subset in subsets:
            if contains_q3(adj, subset):
                is_free = False
                found_q3_subset = subset
                # Find edges in this subset to remove
                # We need to find a Q3 subgraph to remove one edge from.
                # For now, just remove a random edge from the subset.
                break
        
        if is_free:
            # Update best
            edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = adj[:]
            
            # Try to add a random edge
            # Pick a random missing edge
            missing_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        missing_edges.append((i, j))
            
            if missing_edges:
                u, v = random.choice(missing_edges)
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if it created a Q3
                # We can just check the subsets containing u and v
                # But for simplicity, check all?
                # No, check only subsets with u and v.
                created_q3 = False
                for subset in subsets:
                    if u in subset and v in subset:
                        if contains_q3(adj, subset):
                            created_q3 = True
                            break
                
                if created_q3:
                    # Remove the edge we just added
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    # Or remove another edge?
                    # For now, just revert.
                else:
                    # Keep the edge
                    pass
            else:
                # Graph is complete, break
                break
        else:
            # Found a Q3, remove an edge from it
            # We need to find the specific edges of the Q3.
            # For simplicity, pick a random edge from the subset.
            edges_in_subset = []
            for i in range(len(found_q3_subset)):
                for j in range(i+1, len(found_q3_subset)):
                    u = found_q3_subset[i]
                    v = found_q3_subset[j]
                    if (adj[u] >> v) & 1:
                        edges_in_subset.append((u, v))
            
            if edges_in_subset:
                u, v = random.choice(edges_in_subset)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Should not happen
                break

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
