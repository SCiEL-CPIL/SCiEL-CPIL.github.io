import random


def construct():
    n = 15
    best_edges = []
    
    def find_invalid_edges(edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        deg = {i: len(adj[i]) for i in range(n)}
        
        invalid = []
        for u, v in edges:
            neighbors_u = [x for x in adj[u] if x != v]
            neighbors_v = [y for y in adj[v] if y != u]
            
            # Map degree to list of vertices
            deg_map_u = {}
            for x in neighbors_u:
                d = deg[x]
                if d not in deg_map_u:
                    deg_map_u[d] = []
                deg_map_u[d].append(x)
                
            deg_map_v = {}
            for y in neighbors_v:
                d = deg[y]
                if d not in deg_map_v:
                    deg_map_v[d] = []
                deg_map_v[d].append(y)
                
            is_conflict = False
            for d in deg_map_u:
                if d in deg_map_v:
                    list_u = deg_map_u[d]
                    list_v = deg_map_v[d]
                    # Conflict if there exists s in list_u, t in list_v such that s != t
                    # This is false only if list_u == list_v == [x] for some x
                    if not (len(list_u) == 1 and len(list_v) == 1 and list_u[0] == list_v[0]):
                        is_conflict = True
                        break
                        
            if is_conflict:
                invalid.append((u, v))
                
        return invalid

    # Try multiple random initializations
    for seed in range(300):
        random.seed(seed)
        current_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Repair loop
        while True:
            invalids = find_invalid_edges(current_edges)
            if not invalids:
                break
            
            # Heuristic: Remove an edge. 
            # Random choice is simple. 
            # We can also try to remove an edge incident to a high-degree vertex to reduce degrees faster.
            # But let's stick to random for robustness, or maybe pick one with max degree sum?
            # For now, random.
            edge_to_remove = random.choice(invalids)
            current_edges.remove(edge_to_remove)
            
        if len(current_edges) > len(best_edges):
            best_edges = current_edges[:]
            
    return [list(e) for e in best_edges]
