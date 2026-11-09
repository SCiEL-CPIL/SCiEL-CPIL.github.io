import numpy as np
import time
import random

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    best_edges = 0
    best_adj = None

    # Helper to count edges
    def count_edges(adj):
        return int(np.sum(adj)) // 2

    # Helper to check K2,3-free
    def is_k23_free(adj):
        common = adj @ adj
        # Exclude diagonal (self-loops not present, but adj@adj has diagonal = degree)
        # We only care about i != j
        mask = ~np.eye(n, dtype=bool)
        return np.all(common[mask] <= 2)

    def valid_add(adj, u, v):
        if adj[u, v]:
            return False
        # 1. common[u][v] must be <= 2
        common_uv = int(adj[u] @ adj[v])
        if common_uv > 2:
            return False
        
        # 2. For any w in N(u), common[w][v] must be <= 1
        #    N(u) is indices where adj[u] is True
        neighbors_u = np.where(adj[u])[0]
        for w in neighbors_u:
            if w == v: continue
            # common[w][v] = adj[w] @ adj[v]
            if int(adj[w] @ adj[v]) > 1:
                return False
                
        # 3. For any w in N(v), common[w][u] must be <= 1
        neighbors_v = np.where(adj[v])[0]
        for w in neighbors_v:
            if w == u: continue
            if int(adj[w] @ adj[u]) > 1:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u, v] = 1
        adj[v, u] = 1

    def remove_edge(adj, u, v):
        adj[u, v] = 0
        adj[v, u] = 0

    # Initialize with a random valid graph
    def random_valid_graph():
        adj = np.zeros((n, n), dtype=int)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if valid_add(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Local Search with Simulated Annealing
    def local_search(adj, T_start, T_min, cooling):
        current_adj = adj.copy()
        current_edges = count_edges(current_adj)
        best_local_adj = current_adj.copy()
        best_local_edges = current_edges
        
        T = T_start
        while T > T_min:
            if time.time() - start_time > time_limit:
                break
                
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random non-edge
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not current_adj[u, v]]
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_adj = current_adj.copy()
                else:
                    # Reject
                    pass
            elif move_type < 0.8:
                # Try to remove a random edge
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if current_adj[u, v]]
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                # Accept with prob exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    remove_edge(current_adj, u, v)
                    current_edges -= 1
            else:
                # Swap: remove one, add one
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if current_adj[u, v]]
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not current_adj[u, v]]
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                
                # Check validity of swap
                # Save state
                old_adj = current_adj.copy()
                remove_edge(current_adj, u1, v1)
                if valid_add(current_adj, u2, v2):
                    add_edge(current_adj, u2, v2)
                    # Edge count same
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_adj = current_adj.copy()
                else:
                    current_adj = old_adj
                    
            T *= cooling
            
        return best_local_adj, best_local_edges

    # Run multiple restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Start with a random graph or a perturbed best graph
        if best_adj is None:
            adj = random_valid_graph()
        else:
            # Perturb best graph
            adj = best_adj.copy()
            # Remove some random edges
            for _ in range(random.randint(2, 5)):
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if edges:
                    u, v = random.choice(edges)
                    remove_edge(adj, u, v)
        
        # Run local search
        adj_res, edges_res = local_search(adj, 1.0, 0.01, 0.999)
        
        if edges_res > best_edges:
            best_edges = edges_res
            best_adj = adj_res
            
    # Extract edges
    if best_adj is None:
        return []
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return edges
