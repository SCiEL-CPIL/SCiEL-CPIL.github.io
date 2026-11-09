import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 15
    # Partition: A = {0,1,2,3,4,5,6}, B = {7,8,9,10,11,12,13,14}
    A = list(range(7))
    B = list(range(7, 15))
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Base graph: K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
            
    # Edges to consider adding: Intra-A and Intra-B
    # We process these in a specific order to maximize density
    intra_A = [(i, j) for i in range(7) for j in range(i+1, 7)]
    intra_B = [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    
    # Combine edges to add, prioritizing those that might be "easier" to add?
    # Or just process all of them.
    # To maximize, we want to include as many as possible.
    # We will use a backtracking search that tries to include first.
    
    # Pre-compute for each vertex, which edges (by index in candidate list) are incident to it
    # We need to map edge tuples to indices in the candidate list
    candidate_edges = intra_A + intra_B
    edge_to_idx = {e: i for i, e in enumerate(candidate_edges)}
    
    # For each vertex, list of candidate edge indices incident to it
    vertex_incident_edges = {v: [] for v in range(n)}
    for i, e in enumerate(candidate_edges):
        u, v = e
        vertex_incident_edges[u].append(i)
        vertex_incident_edges[v].append(i)
        
    # Sort incident edges by index (processing order)
    for v in range(n):
        vertex_incident_edges[v].sort()

    def is_possibly_valid(edge_idx, current_edge_set, adj, deg):
        """
        Check if the current state can lead to a valid graph.
        Returns True if no unavoidable violation exists.
        """
        # Check all 3-paths in the current graph
        # A 3-path is s-u-v-t.
        # We only need to check paths that involve at least one edge added in the current step?
        # No, adding an edge can create new paths.
        # However, if a path existed before and was valid (deg(s) != deg(t)), 
        # and degrees haven't changed for s,t, it's still valid.
        # If degrees changed, we need to re-check.
        
        # To be safe and fast, we check all 3-paths in the current graph.
        # n=15, so this is fast.
        
        for s in range(n):
            for u in adj[s]:
                if u == s: continue
                for v in adj[u]:
                    if v == s or v == u: continue
                    for t in adj[v]:
                        if t == s or t == u or t == v: continue
                        # Path s-u-v-t
                        if deg[s] == deg[t]:
                            # Violation found in current graph.
                            # Can it be fixed?
                            # deg[s] can increase if there are unprocessed edges incident to s.
                            # deg[t] can increase if there are unprocessed edges incident to t.
                            # If neither can increase, the violation is permanent.
                            
                            # Check if s has any unprocessed incident edges
                            s_can_change = False
                            for e_idx in vertex_incident_edges[s]:
                                if e_idx > edge_idx:
                                    s_can_change = True
                                    break
                                    
                            t_can_change = False
                            for e_idx in vertex_incident_edges[t]:
                                if e_idx > edge_idx:
                                    t_can_change = True
                                    break
                                    
                            if not s_can_change and not t_can_change:
                                return False
                            
        return True

    def search(edge_idx, current_edge_set, adj, deg):
        """
        Recursive backtracking search.
        Returns the set of edges for the best valid graph found from this state.
        """
        if time.time() > deadline:
            return current_edge_set # Return current best effort? No, must be valid.
            # Actually, if we timeout, we should return a known valid graph.
            # We'll handle timeout at the top level.
            
        if edge_idx == len(candidate_edges):
            # All edges processed. Check if valid.
            # is_possibly_valid checks for unavoidable violations. 
            # If we reach the end, all degrees are fixed.
            # So if is_possibly_valid returns True, it means no violations exist?
            # Yes, because if there was a violation, s_can_change and t_can_change would be False, 
            # and it would have returned False earlier.
            # So if we return True here, the graph is valid.
            return current_edge_set

        # Try to INCLUDE the edge
        u, v = candidate_edges[edge_idx]
        
        # Optimization: If u and v are already connected? No, simple graph.
        
        new_edge_set = current_edge_set | {(u, v)}
        new_adj = [set(x) for x in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        new_deg = deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        
        if is_possibly_valid(edge_idx, new_edge_set, new_adj, new_deg):
            # Recurse
            result_include = search(edge_idx + 1, new_edge_set, new_adj, new_deg)
            if result_include is not None:
                # Since we are maximizing, and we added an edge, this is better than excluding
                # provided the result is valid.
                # The search returns a valid set.
                return result_include
        
        # Try to EXCLUDE the edge
        result_exclude = search(edge_idx + 1, current_edge_set, adj, deg)
        return result_exclude

    # Start the search
    # Initialize with base graph
    init_adj = [set() for _ in range(n)]
    init_deg = [0] * n
    for u, v in base_edges:
        init_adj[u].add(v)
        init_adj[v].add(u)
        init_deg[u] += 1
        init_deg[v] += 1
        
    # The search tries to add all candidate edges.
    # It returns the set of added edges that are valid.
    # We need to combine base_edges with the result.
    
    # To avoid recursion depth issues (15+15=30 depth is fine), 
    # and to handle timeout, we wrap in a try/except or check time.
    
    best_edges = set(base_edges)
    
    try:
        added_edges = search(0, set(), init_adj, init_deg)
        if added_edges is not None:
            best_edges = base_edges | added_edges
    except Exception:
        pass

    # If the search didn't find anything or timed out, best_edges is base_edges (56)
    # We can try a simple local search to improve if we have time.
    
    if time.time() < deadline - 5:
        # Local search on best_edges
        # Try to add any remaining candidate edge that wasn't added
        remaining = [e for e in candidate_edges if e not in best_edges]
        random.shuffle(remaining)
        
        for e in remaining:
            if time.time() > deadline - 2:
                break
            test_edges = best_edges | {e}
            # Quick check
            adj = [set() for _ in range(n)]
            deg = [0] * n
            for u, v in test_edges:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
            
            valid = True
            for s in range(n):
                for u in adj[s]:
                    for v in adj[u]:
                        if v == s: continue
                        for t in adj[v]:
                            if t == s or t == u: continue
                            if deg[s] == deg[t]:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            
            if valid:
                best_edges = test_edges

    # Final validation
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
    valid = True
    for s in range(n):
        for u in adj[s]:
            for v in adj[u]:
                if v == s: continue
                for t in adj[v]:
                    if t == s or t == u: continue
                    if deg[s] == deg[t]:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to base
        best_edges = base_edges

    result = []
    for u, v in sorted(best_edges):
        result.append([int(u), int(v)])
    
    return result
