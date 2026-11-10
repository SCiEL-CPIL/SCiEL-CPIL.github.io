import random
import numpy as np

def construct():
    n = 20
    # Use a fixed seed for reproducibility in the function, 
    # but the task implies the function is called multiple times or just once.
    # To maximize score, we should find the best possible graph.
    # We will run a local search.
    
    def is_valid(adj_matrix):
        # Check if any pair has > 2 common neighbors
        # adj_matrix is n x n boolean
        for u in range(n):
            for v in range(u + 1, n):
                common = np.sum(adj_matrix[u] & adj_matrix[v])
                if common > 2:
                    return False
        return True

    def get_edges(adj_matrix):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_matrix[u, v]:
                    edges.append([u, v])
        return edges

    best_edges = 0
    best_graph = None
    
    # Try multiple random restarts
    for _ in range(100):
        # Initialize with random graph
        adj = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        # Local search
        improved = True
        while improved:
            improved = False
            # Check validity
            if not is_valid(adj):
                # Find violation and fix
                for u in range(n):
                    for v in range(u + 1, n):
                        common = np.sum(adj[u] & adj[v])
                        if common > 2:
                            # Find a neighbor to remove
                            # The common neighbors are the intersection
                            # We need to break the K2,3.
                            # Remove an edge (u, w) or (v, w) where w is common neighbor?
                            # Actually, removing (u, w) reduces common neighbors of (u, w) and (v, w) etc.
                            # Let's just remove a random edge incident to u or v.
                            # Better: remove an edge that is part of the violation.
                            # The violation is that u and v share 3 neighbors.
                            # Let w1, w2, w3 be common neighbors.
                            # If we remove (u, w1), then u and v share 2 neighbors.
                            # But u and w1 might now have issues? No, removing edge only decreases counts.
                            # So removing any edge incident to u or v that connects to a common neighbor fixes this pair.
                            # It might create other issues? No, removing edges never creates K2,3.
                            # So we can safely remove an edge.
                            common_neighbors = np.where(adj[u] & adj[v])[0]
                            w = common_neighbors[random.randint(0, len(common_neighbors) - 1)]
                            # Remove edge (u, w) or (v, w). Let's pick u.
                            adj[u, w] = 0
                            adj[w, u] = 0
                            improved = True
                            break
                    if improved:
                        break
            else:
                # Valid, try to add edge
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v] == 0:
                            non_edges.append((u, v))
                
                if not non_edges:
                    break
                
                # Shuffle to avoid bias
                random.shuffle(non_edges)
                added = False
                for u, v in non_edges:
                    # Check if adding (u, v) keeps it valid
                    # New common neighbors for (u, v) will be 0? No.
                    # Adding (u, v) increases common neighbor count for pairs (u, x) and (v, x) where x is neighbor of both?
                    # No.
                    # If we add (u, v), the common neighbors of u and v become the intersection of their neighbor sets.
                    # Let C = N(u) intersect N(v).
                    # We need |C| <= 2.
                    # Also, for any x in N(u), the common neighbors of u and x are N(u) intersect N(x).
                    # Since v is now in N(u), if v is in N(x), then v becomes a new common neighbor of u and x.
                    # So we need:
                    # 1. |N(u) intersect N(v)| <= 2
                    # 2. For all x in N(u) \ {v}, |N(u) intersect N(x)| + (1 if v in N(x) else 0) <= 2
                    #    Since graph was valid, |N(u) intersect N(x)| <= 2.
                    #    If v in N(x), then new count is old_count + 1.
                    #    So we need old_count <= 1 for all x in N(u) such that v in N(x).
                    #    i.e., if v is neighbor of x, then u and x must have had <= 1 common neighbor before.
                    # 3. Similarly for y in N(v) \ {u}, if u in N(y), then v and y must have had <= 1 common neighbor.
                    
                    # Let's just simulate the addition on a copy for simplicity, n=20 is small.
                    adj_temp = adj.copy()
                    adj_temp[u, v] = 1
                    adj_temp[v, u] = 1
                    if is_valid(adj_temp):
                        adj = adj_temp
                        improved = True
                        added = True
                        break
                
                if not added:
                    # Stuck, remove a random edge to escape local optimum
                    edges = get_edges(adj)
                    if edges:
                        e = random.choice(edges)
                        adj[e[0], e[1]] = 0
                        adj[e[1], e[0]] = 0
                        improved = True
                    else:
                        break
        
        current_edges = np.sum(adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = adj.copy()
            
    return get_edges(best_graph)
