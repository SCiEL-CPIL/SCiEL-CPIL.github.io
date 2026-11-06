import random
import time

def construct():
    """
    Constructs a K3,3-free graph on 20 vertices maximizing edges.
    Uses greedy construction with random restarts and local search.
    """
    N = 20
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    best_edge_count = 0
    best_adj = [0] * N
    
    def popcount(x):
        return bin(x).count("1")
        
    def is_valid_add(adj, u, v):
        """
        Checks if adding edge (u, v) to graph 'adj' keeps it K3,3-free.
        Assumes u and v are not currently connected.
        """
        S_A_mask = adj[v] 
        S_B_mask = adj[u] 
        
        verts_A = [i for i in range(N) if (S_A_mask >> i) & 1]
        
        for i in range(len(verts_A)):
            a1 = verts_A[i]
            for j in range(i + 1, len(verts_A)):
                a2 = verts_A[j]
                
                if (S_B_mask >> a1) & 1:
                    continue
                if (S_B_mask >> a2) & 1:
                    continue
                    
                common = adj[a1] & adj[a2]
                common_in_SB = common & S_B_mask
                
                if popcount(common_in_SB) >= 2:
                    return False
                    
        return True

    def run_local_search(adj, edge_count):
        """
        Attempts to improve the graph by adding edges.
        Returns new adj and edge_count.
        """
        improved = True
        # Limit local search steps to prevent getting stuck in long loops
        # if it doesn't converge quickly, though for N=20 it should be fine.
        max_local_iters = 50
        local_iter = 0
        
        while improved and local_iter < max_local_iters:
            improved = False
            local_iter += 1
            # Shuffle to avoid bias? Or just iterate.
            # Iterating in fixed order is fine for local search improvement.
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_count += 1
                        improved = True
        return adj, edge_count

    # Main loop: Random Restart Greedy
    while time.time() < deadline:
        # Reset graph
        adj = [0] * N
        edge_count = 0
        
        # Random order of edges
        random.shuffle(all_edges)
        
        # Greedy phase
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    
        # Local search phase
        adj, edge_count = run_local_search(adj, edge_count)
        
        # Update best
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = adj[:]
            
    # Construct result list
    result = []
    for u in range(N):
        mask = best_adj[u]
        for v in range(u + 1, N):
            if (mask >> v) & 1:
                result.append([u, v])
                
    return result
