import random
import itertools
import time

def construct():
    n = 20
    best_edges_count = 0
    best_edge_list = []
    
    # Precompute all pairs of vertices for randomization
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(adj, u, v):
        """
        Check if adding edge (u, v) to graph adj creates a K3,3.
        Returns True if safe, False if it creates a K3,3.
        """
        # Get neighbors of u and v in current graph
        # Note: adj[u][v] is currently False
        N_u = [x for x in range(n) if adj[u][x]]
        N_v = [x for x in range(n) if adj[v][x]]
        
        # To form a K3,3 with (u,v), we need:
        # u in Part A, v in Part B.
        # We need to choose 2 more vertices from A (neighbors of v) and 2 more from B (neighbors of u).
        # Let A_extra be a pair from N_v.
        # Let B_extra be a pair from N_u.
        # The edges (a, b) for a in A_extra, b in B_extra must exist.
        
        # Optimization: If |N_u| < 2 or |N_v| < 2, it's impossible to form K3,3 with u in A, v in B.
        # But we also need to consider u in B, v in A.
        # Actually, the check is symmetric. We just need to check if there exist
        # x1, x2 in N(v) and y1, y2 in N(u) such that
        # adj[x1][y1], adj[x1][y2], adj[x2][y1], adj[x2][y2] are all True.
        
        if len(N_u) < 2 or len(N_v) < 2:
            return True
            
        # Iterate over all pairs in N_v (candidates for A_extra)
        # And all pairs in N_u (candidates for B_extra)
        # To speed up, we can precompute for each pair in N_v, the common neighbors in N_u?
        # Or just nested loops. n=20 is small.
        
        for x1, x2 in itertools.combinations(N_v, 2):
            # Check if there are at least 2 common neighbors of x1 and x2 within N_u
            common_count = 0
            # We can break early if we find 2
            for y in N_u:
                if adj[x1][y] and adj[x2][y]:
                    common_count += 1
                    if common_count >= 2:
                        return False
            # If we didn't find 2, continue to next pair
            
        return True

    def greedy_build(seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        
        pairs = all_pairs.copy()
        random.shuffle(pairs)
        
        edge_count = 0
        for u, v in pairs:
            if can_add(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
                edge_count += 1
                
        return edge_count, adj

    start_time = time.time()
    max_time = 100  # Leave some buffer for 120s limit
    
    # Try multiple seeds
    # Since the problem is small, we can try many restarts
    for seed in range(1000):
        if time.time() - start_time > max_time:
            break
            
        count, adj = greedy_build(seed)
        
        if count > best_edges_count:
            best_edges_count = count
            best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
            
            # If we find a very good solution, we might want to try local search?
            # But the directive says "design a new program... from scratch".
            # A pure greedy with restarts is a distinct approach.
            # Let's add a simple local search (swap/flip) if time permits?
            # No, let's stick to the core new approach: Randomized Greedy Construction.
            # We can improve the greedy by trying different orderings or using a priority queue.
            # But simple random shuffle restarts are robust.
            
            # Optional: Try to add back edges that were skipped?
            # The greedy algorithm processes edges in random order.
            # Sometimes an edge is rejected because it creates a K3,3, but if we had added a different set of edges, it might have been safe.
            # A "repair" phase could help.
            # However, for n=20, the greedy is quite strong.
            
    return best_edge_list
