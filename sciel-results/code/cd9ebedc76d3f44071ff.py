import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        # Clique on first k
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # 1. Construct cross cliques
        cliques = []
        covered_edges = set()
        
        # Adjacency list for C to I
        # N(v) for v in I
        neighbors_in_C = {v: set() for v in range(k, n)}
        for (u, v) in edges:
            if u < k and v >= k:
                neighbors_in_C[v].add(u)
            elif v < k and u >= k:
                neighbors_in_C[u].add(v)
        
        for v in range(k, n):
            N_v = neighbors_in_C[v]
            if len(N_v) > 0:
                clique = [v] + sorted(list(N_v))
                cliques.append(clique)
                # Mark edges as covered
                # Cross edges (v, u)
                for u in N_v:
                    covered_edges.add((min(v, u), max(v, u)))
                # Internal edges (u1, u2)
                N_v_list = sorted(list(N_v))
                for i in range(len(N_v_list)):
                    for j in range(i + 1, len(N_v_list)):
                        u1 = N_v_list[i]
                        u2 = N_v_list[j]
                        covered_edges.add((u1, u2))
            
            # If N_v is empty, v is isolated in the split sense.
            # No clique needed.
            
        # 2. Handle remaining internal edges
        remaining_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_edges:
                    remaining_edges.add((u, v))
        
        # Build adjacency for remaining graph on C
        rem_adj = {i: set() for i in range(k)}
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Greedy Max Clique Partition
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add each remaining edge as a clique
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Pick a vertex with highest degree
            # Filter out vertices with 0 degree to avoid empty cliques
            active_vertices = [v for v in rem_adj if len(rem_adj[v]) > 0]
            if not active_vertices:
                break # Should not happen if remaining_edges is not empty
                
            u = max(active_vertices, key=lambda x: len(rem_adj[x]))
            candidates = list(rem_adj[u])
            
            # Greedy clique finding
            clique = [u]
            # Sort candidates by degree to prioritize high-degree nodes
            candidates.sort(key=lambda x: len(rem_adj[x]), reverse=True)
            
            for v in candidates:
                # Check if v is connected to all in clique
                is_connected = True
                for clique_node in clique:
                    e = (min(clique_node, v), max(clique_node, v))
                    if e not in remaining_edges:
                        is_connected = False
                        break
                if is_connected:
                    clique.append(v)
            
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                
                # Remove edges in clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                            rem_adj[clique[i]].remove(clique[j])
                            rem_adj[clique[j]].remove(clique[i])
            else:
                # If we couldn't form a clique of size 2, something is wrong
                # But if remaining_edges is not empty, u has a neighbor.
                # So this shouldn't happen.
                # Force add an edge to avoid infinite loop
                if remaining_edges:
                    e = next(iter(remaining_edges))
                    cliques.append(list(e))
                    remaining_edges.remove(e)
                    rem_adj[e[0]].remove(e[1])
                    rem_adj[e[1]].remove(e[0])
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # If we are running out of time, we can just return the seed result for the remaining instances
            # But we must return a valid result.
            # The seed result is valid.
            # However, we should try to solve if possible.
            pass
        
        res = solve_instance(*spec)
        # Convert to lists of lists of ints
        final_res = []
        for c in res:
            final_res.append([int(x) for x in c])
        results.append(final_res)
        
    return results
