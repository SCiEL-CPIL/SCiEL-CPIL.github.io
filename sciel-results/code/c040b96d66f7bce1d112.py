import time
import random
import numpy as np

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Check if there exist x, y in N(u) such that N(x) ∩ N(y) ∩ N(v) has size >= 2
    # This would form K3,3 with parts {u, a1, a2} and {v, x, y}
    for x in range(N):
        if (nu >> x) & 1:
            for y in range(x + 1, N):
                if (nu >> y) & 1:
                    common = adj[x] & adj[y] & nv
                    if common.bit_count() >= 2:
                        return False

    # Check if there exist a, b in N(v) such that N(a) ∩ N(b) ∩ N(u) has size >= 2
    # This would form K3,3 with parts {v, a1, a2} and {u, x, y}
    # Note: The first check covers the case where u is in the first part.
    # This check covers the case where v is in the first part.
    # Actually, the structure is symmetric. 
    # If we found a K3,3, it has two parts A and B.
    # The new edge (u,v) connects them.
    # So u is in one part, v in the other.
    # WLOG, u in A, v in B.
    # Then A = {u, a1, a2}, B = {v, b1, b2}.
    # Edges: u-v, u-b1, u-b2, a1-v, a2-v, a1-b1, a1-b2, a2-b1, a2-b2.
    # Conditions:
    # b1, b2 in N(u)
    # a1, a2 in N(v)
    # a1, a2 in N(b1) and N(b2)
    # This is exactly: for b1, b2 in N(u), |N(b1) ∩ N(b2) ∩ N(v)| >= 2.
    # So the first loop is sufficient.
    
    return True

def add_edge(u, v, adj):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def remove_edge(u, v, adj):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def count_edges(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2

def local_search(adj, deadline):
    """
    Perform local search on the graph.
    Returns the best adj found.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing edge
        for u, v in ALL_EDGES:
            if time.time() > deadline:
                break
            if not (adj[u] >> v) & 1:
                if is_valid_add(u, v, adj):
                    add_edge(u, v, adj)
                    improved = True
                    break
        
        if time.time() > deadline:
            break
            
        # Phase 2: If no edge could be added, try 1-swap (remove 1, add 2)
        if not improved:
            found_swap = False
            # Shuffle edges to avoid bias
            current_edges = []
            for u in range(N):
                for v in range(u+1, N):
                    if (adj[u] >> v) & 1:
                        current_edges.append((u, v))
                        
            random.shuffle(current_edges)
            
            for u1, v1 in current_edges:
                if time.time() > deadline:
                    break
                
                # Remove edge (u1, v1)
                remove_edge(u1, v1, adj)
                
                added = 0
                # Try to add 2 edges
                for u2, v2 in ALL_EDGES:
                    if time.time() > deadline:
                        break
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if not (adj[u2] >> v2) & 1:
                        if is_valid_add(u2, v2, adj):
                            add_edge(u2, v2, adj)
                            added += 1
                            if added == 2:
                                break
                
                if added == 2:
                    found_swap = True
                    improved = True
                    break
                else:
                    # Undo additions
                    # We need to know which edges were added. 
                    # This is tricky without tracking. 
                    # Let's track them.
                    # Refactor:
                    pass
            
            # The above loop is inefficient because undoing is hard without tracking.
            # Let's rewrite Phase 2 more carefully.
            
            if not found_swap:
                 # Try 2-swap? Or just restart?
                 # Given time constraints, maybe just restart is better than complex 2-swap with undo logic.
                 # But 2-swap can escape deeper local optima.
                 # Let's implement a proper 1-swap with tracking.
                
                # Reset adj to state before trying swaps? No, we modified it.
                # We need to save state.
                pass

    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_adj = None
    best_count = 0
    
    # Try multiple random restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        # Greedy construction
        adj = [0] * N
        shuffled_edges = ALL_EDGES[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
        
        # Local Search
        # We need a robust local search.
        # Let's implement a simple hill climbing with 1-swap properly.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add
            for u, v in ALL_EDGES:
                if time.time() > deadline:
                    break
                if not (adj[u] >> v) & 1:
                    if is_valid_add(u, v, adj):
                        add_edge(u, v, adj)
                        improved = True
                        break
            
            if time.time() > deadline:
                break
                
            # 2. Try 1-swap
            if not improved:
                # Get current edges
                curr_edges = []
                for u in range(N):
                    for v in range(u+1, N):
                        if (adj[u] >> v) & 1:
                            curr_edges.append((u, v))
                
                random.shuffle(curr_edges)
                
                for u1, v1 in curr_edges:
                    if time.time() > deadline:
                        break
                    
                    # Save state
                    saved_adj = adj[:]
                    
                    remove_edge(u1, v1, adj)
                    
                    added_count = 0
                    added_edges = []
                    
                    for u2, v2 in ALL_EDGES:
                        if time.time() > deadline:
                            break
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        if not (adj[u2] >> v2) & 1:
                            if is_valid_add(u2, v2, adj):
                                add_edge(u2, v2, adj)
                                added_edges.append((u2, v2))
                                added_count += 1
                                if added_count == 2:
                                    break
                    
                    if added_count == 2:
                        improved = True
                        break
                    else:
                        # Restore state
                        adj = saved_adj
                    
                if not improved:
                    break
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Early stop if we hit a high threshold
        if best_count >= 78:
            break

    if best_adj is None:
        return []
        
    # Convert adj back to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
