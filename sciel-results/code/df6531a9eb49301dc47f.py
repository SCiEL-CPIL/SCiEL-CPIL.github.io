import random
from itertools import combinations

def construct():
    n = 18
    vertices = list(range(n))
    
    def is_k4_free(adj, edges):
        """Check if graph is K4-free using adjacency matrix."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    # Check if u,v form a K4 with any two other vertices
                    common = set()
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            common.add(w)
                    # Check if any pair in common is connected
                    common_list = list(common)
                    for i in range(len(common_list)):
                        for j in range(i+1, len(common_list)):
                            if adj[common_list[i]][common_list[j]]:
                                return False
        return True
    
    def can_add_edge(adj, edges, u, v):
        """Check if adding edge (u,v) maintains K4-freeness."""
        # Common neighbors of u and v
        common = set()
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common.add(w)
        
        # If there are two common neighbors that are connected, adding (u,v) creates K4
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                if adj[common_list[i]][common_list[j]]:
                    return False
        return True
    
    def max_triangle_free_size(adj):
        """Find the largest triangle-free induced subgraph."""
        # Check from size 12 down to 1
        for size in range(12, 0, -1):
            for combo in combinations(vertices, size):
                # Check if this subset is triangle-free
                is_tf = True
                for i in range(len(combo)):
                    if not is_tf:
                        break
                    for j in range(i+1, len(combo)):
                        if not is_tf:
                            break
                        if adj[combo[i]][combo[j]]:
                            for k in range(j+1, len(combo)):
                                if adj[combo[i]][combo[k]] and adj[combo[j]][combo[k]]:
                                    is_tf = False
                                    break
                if is_tf:
                    return size
        return 0
    
    best_score = 18
    best_edges = None
    
    # Multiple restarts with local search
    for trial in range(30):
        random.seed(trial * 12345 + 67890)
        
        # Start with empty graph and add edges greedily to make it dense but K4-free
        adj = [[0]*n for _ in range(n)]
        edges = set()
        
        # Random order of edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        # Add edges greedily
        for u, v in all_pairs:
            if can_add_edge(adj, edges, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                edges.add((u, v))
        
        # Local search: try to improve by adding edges that break large triangle-free sets
        score = max_triangle_free_size(adj)
        
        # Iteratively try to add more edges
        for iteration in range(100):
            # Find a large triangle-free set
            tf_set = None
            for size in range(score, 0, -1):
                for combo in combinations(vertices, size):
                    is_tf = True
                    for i in range(len(combo)):
                        if not is_tf:
                            break
                        for j in range(i+1, len(combo)):
                            if not is_tf:
                                break
                            if adj[combo[i]][combo[j]]:
                                for k in range(j+1, len(combo)):
                                    if adj[combo[i]][combo[k]] and adj[combo[j]][combo[k]]:
                                        is_tf = False
                                        break
                    if is_tf:
                        tf_set = list(combo)
                        break
                if tf_set:
                    break
            
            if not tf_set:
                break
            
            # Try to add an edge within this triangle-free set that is K4-safe
            added = False
            for i in range(len(tf_set)):
                if added:
                    break
                for j in range(i+1, len(tf_set)):
                    u, v = tf_set[i], tf_set[j]
                    if not adj[u][v] and can_add_edge(adj, edges, u, v):
                        # Add this edge
                        adj[u][v] = 1
                        adj[v][u] = 1
                        edges.add((u, v))
                        added = True
                        break
            
            if not added:
                # Try removing some edges and re-adding differently
                # Or just break
                break
            
            # Update score
            new_score = max_triangle_free_size(adj)
            if new_score < score:
                score = new_score
                if score < best_score:
                    best_score = score
                    best_edges = list(edges)
                if score <= 11:
                    return [[u, v] for u, v in best_edges]
        
        if score < best_score:
            best_score = score
            best_edges = list(edges)
    
    return [[u, v] for u, v in best_edges]
