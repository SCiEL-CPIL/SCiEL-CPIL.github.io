"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    # Adjacency matrix for fast lookup
    # adj[u][v] is 1 if edge exists
    adj = [[0] * n for _ in range(n)]
    edges = []
    
    # Precompute potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to get common neighbors count for u, v
    def common_neighbors(u, v):
        count = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                count += 1
        return count
    
    # Helper to check if adding edge (u, v) is valid
    # Valid if common_neighbors(u, v) < 2 (so that adding it makes it <= 2)
    # Wait, the constraint is "no two vertices have three common neighbours".
    # So for any pair (u, v), |N(u) ∩ N(v)| <= 2.
    # If we add edge (u, v), it does NOT change N(u) ∩ N(v) for the pair (u, v) themselves?
    # No, N(u) is the set of neighbors. Adding edge (u, v) adds v to N(u) and u to N(v).
    # Does this affect the common neighbor count for other pairs?
    # Let's check:
    # For a pair (x, y), their common neighbors are w such that adj[x][w] and adj[y][w].
    # Adding edge (u, v) sets adj[u][v]=1 and adj[v][u]=1.
    # This changes common neighbor counts for pairs involving u or v.
    # Specifically, for pair (u, w) where w != v:
    #   N(u) gains v. So if adj[w][v] is 1, then v becomes a common neighbor of u and w.
    #   So |N(u) ∩ N(w)| increases by 1 if adj[w][v] == 1.
    #   We need new count <= 2. So old count + (1 if adj[w][v] else 0) <= 2.
    #   i.e., old_count <= 2 - (1 if adj[w][v] else 0).
    #   If adj[w][v] is 1, old_count must be <= 1.
    #   If adj[w][v] is 0, old_count must be <= 2.
    # Similarly for pair (v, w).
    # And for pair (u, v) itself:
    #   The edge (u, v) is added. u is in N(v), v is in N(u).
    #   Do u and v become common neighbors of each other? No, common neighbors are OTHER vertices.
    #   But does adding the edge change |N(u) ∩ N(v)|?
    #   N(u) gains v. N(v) gains u.
    #   Intersection N(u) ∩ N(v) consists of w such that w in N(u) and w in N(v).
    #   v is in N(u) but is v in N(v)? No (simple graph, no loops).
    #   u is in N(v) but is u in N(u)? No.
    #   So adding edge (u, v) does NOT change the set of common neighbors of u and v.
    #   It only changes the common neighbor counts for pairs (u, w) and (v, w) for w != u, v.
    
    def can_add_edge(u, v):
        # Check pair (u, w) for all w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
            # For pair (u, w):
            # New common neighbors = Old common neighbors + (1 if v is common neighbor of u and w)
            # v is common neighbor of u and w if adj[u][v] (which will be 1) and adj[w][v] (current).
            # So if adj[w][v] is 1, count increases by 1.
            # We need new_count <= 2.
            # So if adj[w][v] is 1, we need current_count(u, w) <= 1.
            # If adj[w][v] is 0, we need current_count(u, w) <= 2.
            
            cnt = common_neighbors(u, w)
            if adj[w][v]:
                if cnt > 1:
                    return False
            else:
                if cnt > 2:
                    return False
            
            # For pair (v, w):
            cnt = common_neighbors(v, w)
            if adj[w][u]:
                if cnt > 1:
                    return False
            else:
                if cnt > 2:
                    return False
        return True

    def add_edge(u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        edges.append([u, v])

    # Greedy Phase
    # Sort edges by some heuristic?
    # Let's just iterate and add if possible.
    # To make it more robust, we can try multiple random orders.
    
    best_edges = []
    best_count = 0
    
    # Try a few random restarts of greedy
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        current_edges = []
        current_adj = [[0] * n for _ in range(n)]
        
        def can_add_edge_local(u, v, adj_mat):
            for w in range(n):
                if w == u or w == v:
                    continue
                cnt = 0
                for x in range(n):
                    if adj_mat[u][x] and adj_mat[w][x]:
                        cnt += 1
                if adj_mat[w][v]:
                    if cnt > 1:
                        return False
                else:
                    if cnt > 2:
                        return False
                
                cnt = 0
                for x in range(n):
                    if adj_mat[v][x] and adj_mat[w][x]:
                        cnt += 1
                if adj_mat[w][u]:
                    if cnt > 1:
                        return False
                else:
                    if cnt > 2:
                        return False
            return True

        def add_edge_local(u, v, adj_mat, edge_list):
            adj_mat[u][v] = 1
            adj_mat[v][u] = 1
            edge_list.append([u, v])

        for u, v in all_edges:
            if time.time() > deadline:
                break
            if can_add_edge_local(u, v, current_adj):
                add_edge_local(u, v, current_adj, current_edges)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [e[:] for e in current_edges]
            
        # Local Search on best_edges
        # Try to swap edges to increase count?
        # Or just try to add more edges?
        # The greedy might get stuck. Let's try a simple local search:
        # Remove a random edge, then try to add two edges?
        # Or just try to add edges that were skipped?
        
        # Let's try a "repair" or "improvement" step on the current best.
        # We will work on a copy of best_adj
        if best_edges:
            imp_adj = [[0] * n for _ in range(n)]
            for u, v in best_edges:
                imp_adj[u][v] = 1
                imp_adj[v][u] = 1
            imp_edges = [e[:] for e in best_edges]
            
            # Try to add any missing edge
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for u, v in all_edges:
                    if imp_adj[u][v]:
                        continue
                    if can_add_edge_local(u, v, imp_adj):
                        add_edge_local(u, v, imp_adj, imp_edges)
                        improved = True
                        if len(imp_edges) > best_count:
                            best_count = len(imp_edges)
                            best_edges = [e[:] for e in imp_edges]
                # If no improvement, try removing an edge and re-adding?
                # This is expensive. Let's stick to adding for now.
                # If stuck, break.
                
    return best_edges
# EVOLVE-BLOCK-END
