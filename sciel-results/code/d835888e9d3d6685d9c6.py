import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 20
    
    # Adjacency matrix for fast lookup
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Helper to check if adding edge (u, v) violates K2,3 constraint
    # A K2,3 exists if there are two vertices u, v that share at least 3 common neighbors.
    # When adding edge (u, v), we need to ensure:
    # 1. For any w != u, v: if w is connected to u, it must not be connected to v in a way that creates a 3rd common neighbor?
    #    Actually, the condition is: |N(u) \cap N(v)| <= 2.
    #    Adding (u, v) increases |N(u) \cap N(v)| by 1 if there are already 2 common neighbors.
    #    It also affects other pairs.
    #    Specifically, adding (u, v) creates a K2,3 if:
    #    - There exists a pair (x, y) such that |N(x) \cap N(y)| becomes 3.
    #    The only pairs affected are those involving u or v.
    #    - Pair (u, v): |N(u) \cap N(v)| must be < 2 before adding.
    #    - Pair (u, w) for w != v: The common neighbors are (N(u) \cap N(w)). Adding v to N(u) means v is a new common neighbor if v is in N(w).
    #      So if v is in N(w), then |N(u) \cap N(w)| increases by 1. It must have been <= 1 before.
    #    - Pair (v, w) for w != u: Symmetric.
    
    def can_add(u, v):
        if u == v:
            return False
        if adj[u, v]:
            return False
        
        # Check pair (u, v)
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv >= 2:
            return False
            
        # Check pairs (u, w) for all w connected to v
        # If w is connected to v, then v becomes a common neighbor of u and w.
        # So we need |N(u) \cap N(w)| <= 1 for all w in N(v).
        # Note: w cannot be u.
        neighbors_v = np.where(adj[v] == 1)[0]
        for w in neighbors_v:
            if w == u:
                continue
            common_uw = np.sum(adj[u] & adj[w])
            # If v is already a neighbor of u? No, we are adding (u,v).
            # If v is a neighbor of w (it is, by loop), then v will be in N(u) \cap N(w) after addition.
            # So the new count is common_uw + 1 (if v not already in intersection) 
            # Actually, v is in N(w). Is v in N(u)? No, not yet.
            # So v is NOT in the current intersection N(u) \cap N(w).
            # After adding (u,v), v IS in N(u) and IS in N(w). So it becomes a common neighbor.
            # Thus new count = common_uw + 1.
            # We need new count <= 2. So common_uw + 1 <= 2 => common_uw <= 1.
            if common_uw > 1:
                return False
                
        # Check pairs (v, w) for all w connected to u
        neighbors_u = np.where(adj[u] == 1)[0]
        for w in neighbors_u:
            if w == v:
                continue
            common_vw = np.sum(adj[v] & adj[w])
            if common_vw > 1:
                return False
                
        return True

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Initialize with empty graph
    # Or start with seed? Empty is safer for greedy, but seed gives head start.
    # Let's try starting empty and greedily adding edges.
    
    # To ensure some progress, we can randomize the order of edge attempts.
    # However, a good heuristic is to try to add edges that connect low-degree nodes first?
    # Or just randomize.
    
    # Let's use a randomized greedy approach with restarts if time permits.
    # But 20 nodes is small. Complete graph has 190 edges.
    
    best_edges_count = 0
    best_adj = np.zeros((n, n), dtype=np.int8)
    
    # Multiple random restarts
    while time.time() < deadline:
        # Initialize random permutation of edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        current_adj = np.zeros((n, n), dtype=np.int8)
        current_edge_count = 0
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if can_add(u, v) and not current_adj[u, v]:
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_edge_count += 1
        
        if current_edge_count > best_edges_count:
            best_edges_count = current_edge_count
            best_adj = current_adj.copy()
            
        # If we found a very good solution, we can try local search
        if current_edge_count > best_edges_count - 5 and time.time() < deadline - 10:
             # Local search: try to swap edges or add missing ones
             # Simple local search: try to add any missing edge
             improved = True
             while improved and time.time() < deadline - 5:
                 improved = False
                 for u in range(n):
                     for v in range(u + 1, n):
                         if not current_adj[u, v]:
                             if can_add(u, v):
                                 current_adj[u, v] = 1
                                 current_adj[v, u] = 1
                                 current_edge_count += 1
                                 best_edges_count = current_edge_count
                                 best_adj = current_adj.copy()
                                 improved = True
                                 break
                     if improved:
                         break

    return get_edges_from_adj(best_adj)
