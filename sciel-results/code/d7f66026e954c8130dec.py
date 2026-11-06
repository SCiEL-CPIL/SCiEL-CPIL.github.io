import time
import random

def is_valid(graph_adj):
    n = len(graph_adj)
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            # Using bitsets or sets for speed
            # For n=20, simple loops are fine
            common = 0
            for w in range(n):
                if w != u and w != v:
                    if graph_adj[u][w] and graph_adj[v][w]:
                        common += 1
                        if common > 2:
                            return False
    return True

def count_edges(graph_adj):
    n = len(graph_adj)
    edges = 0
    for u in range(n):
        for v in range(u + 1, n):
            if graph_adj[u][v]:
                edges += 1
    return edges

def construct():
    n = 20
    # Deadline: 10 seconds to be safe
    start_time = time.time()
    deadline = start_time + 10.0
    
    best_edges = 0
    best_adj = None
    
    # Seed graph: 5 disjoint K4s
    seed_adj = [[0]*n for _ in range(n)]
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                seed_adj[u][v] = 1
                seed_adj[v][u] = 1
    
    best_adj = [row[:] for row in seed_adj]
    best_edges = count_edges(best_adj)
    
    # Local search function
    def local_search(adj, max_iters=1000):
        current_adj = [row[:] for row in adj]
        current_edges = count_edges(current_adj)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try adding edges
            added = True
            while added:
                added = False
                for u in range(n):
                    if time.time() > deadline - 1.0:
                        return current_adj, current_edges
                for v in range(u + 1, n):
                    if not current_adj[u][v]:
                        # Try adding (u, v)
                        current_adj[u][v] = 1
                        current_adj[v][u] = 1
                        if is_valid(current_adj):
                            current_edges += 1
                            added = True
                            improved = True
                            break # Restart scan from beginning? Or continue?
                            # Continuing might be faster, but restarting ensures we catch new opportunities immediately
                        else:
                            current_adj[u][v] = 0
                            current_adj[v][u] = 0
                    if added:
                        break
                if added:
                    break
            
            if not improved:
                # Try swap: remove one edge, see if we can add more
                # We want to remove an edge that allows adding >= 2 edges
                # Or just remove one and greedily add.
                # To be efficient, we can try removing each edge one by one.
                for u in range(n):
                    if time.time() > deadline - 1.0:
                        return current_adj, current_edges
                    for v in range(u + 1, n):
                        if current_adj[u][v]:
                            # Remove (u, v)
                            current_adj[u][v] = 0
                            current_adj[v][u] = 0
                            temp_edges = current_edges - 1
                            
                            # Try to add as many edges as possible
                            added_any = True
                            while added_any:
                                added_any = False
                                for a in range(n):
                                    if time.time() > deadline - 1.0:
                                        # Restore and return
                                        current_adj[u][v] = 1
                                        current_adj[v][u] = 1
                                        return current_adj, current_edges
                                    for b in range(a + 1, n):
                                        if not current_adj[a][b]:
                                            current_adj[a][b] = 1
                                            current_adj[b][a] = 1
                                            if is_valid(current_adj):
                                                temp_edges += 1
                                                added_any = True
                                                improved = True
                                                break
                                            else:
                                                current_adj[a][b] = 0
                                                current_adj[b][a] = 0
                                        if added_any:
                                            break
                                    if added_any:
                                        break
                            
                            if temp_edges > current_edges:
                                current_edges = temp_edges
                                improved = True
                                break # Accept swap, restart outer loop
                            else:
                                # Restore
                                current_adj[u][v] = 1
                                current_adj[v][u] = 1
                        if improved:
                            break
                    if improved:
                        break
                        
        return current_adj, current_edges

    # Run multiple restarts
    # 1. Start with seed
    adj, edges = local_search(best_adj)
    if edges > best_edges:
        best_edges = edges
        best_adj = adj

    # 2. Random restarts
    for _ in range(20): # Max 20 restarts
        if time.time() > deadline - 2.0:
            break
            
        # Random initial graph
        # Probability p such that expected edges ~ 40
        # n(n-1)/2 = 190. 40/190 ~ 0.2
        p = 0.2 
        rand_adj = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    rand_adj[u][v] = 1
                    rand_adj[v][u] = 1
        
        # Ensure validity of initial random graph?
        # If not valid, we can try to fix it by removing edges, or just let local search handle it?
        # Local search only adds edges. It doesn't remove unless in swap mode.
        # If initial graph is invalid, it stays invalid.
        # So we must ensure initial graph is valid.
        # Simple way: build it by adding edges one by one if valid.
        valid_rand_adj = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    valid_rand_adj[u][v] = 1
                    valid_rand_adj[v][u] = 1
                    if not is_valid(valid_rand_adj):
                        valid_rand_adj[u][v] = 0
                        valid_rand_adj[v][u] = 0
        
        adj, edges = local_search(valid_rand_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj

    # Convert best_adj to list of edges
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges_list.append([u, v])
    return edges_list
