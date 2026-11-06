import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [0]*n
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def largest_clique(adj, mask, current_clique, best_clique):
        # mask is a bitmask of candidate vertices
        # adj is a list of bitmasks
        if not mask:
            if len(current_clique) > len(best_clique):
                best_clique = current_clique[:]
            return best_clique
            
        # Pruning
        if len(current_clique) + bin(mask).count("1") <= len(best_clique):
            return best_clique
            
        # Pick a vertex from mask
        # Heuristic: pick vertex with highest degree in mask
        # We can just iterate
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            new_mask = mask & adj[v]
            new_clique = current_clique + [v]
            best_clique = largest_clique(adj, new_mask, new_clique, best_clique)
            
        return best_clique

    def greedy_partition(n, p, seed):
        adj, edges = random_graph(n, p, seed)
        remaining_edges = set(edges)
        partition = []
        
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        while remaining_edges:
            if time.time() > deadline:
                break
                
            # Pick vertex with max degree
            v = max(range(n), key=lambda x: deg[x])
            if deg[v] == 0:
                break
                
            # Neighbors of v in remaining graph
            # We need to check if edge (v, u) is in remaining_edges
            # This is slow. Better to maintain adjacency as sets of remaining neighbors.
            
        # Let's switch to a different representation for remaining graph
        # rem_adj[u] = set of neighbors v such that (u,v) is in remaining_edges
        
        rem_adj = [set() for _ in range(n)]
        for u, v in edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        remaining_edges_count = len(edges)
        
        while remaining_edges_count > 0:
            if time.time() > deadline:
                break
                
            v = max(range(n), key=lambda x: len(rem_adj[x]))
            if not rem_adj[v]:
                break
                
            neighbors = list(rem_adj[v])
            
            # Find largest clique in neighbors
            # Build adj_mask for the subgraph induced by neighbors
            # We can use the global adj bitmasks, but filtered by neighbors
            
            # Let's use a simple recursive search on the list of neighbors
            # To speed up, we can use the bitmask version on the induced subgraph
            
            # Map neighbors to 0..k-1
            k = len(neighbors)
            if k == 0: break
            idx_map = {v: i for i, v in enumerate(neighbors)}
            
            sub_adj = [0]*k
            for i in range(k):
                for j in range(i+1, k):
                    if neighbors[j] in rem_adj[neighbors[i]]:
                        sub_adj[i] |= (1 << j)
                        sub_adj[j] |= (1 << i)
                        
            best_clique_indices = largest_clique(sub_adj, (1<<k)-1, [], [])
            best_clique = [neighbors[i] for i in best_clique_indices]
            
            if not best_clique:
                # No edge in neighbors?
                # Then we can only form a clique of size 2
                if neighbors:
                    # Pick neighbor with max degree
                    u = max(neighbors, key=lambda x: len(rem_adj[x]))
                    best_clique = [u]
                else:
                    break
                    
            clique = best_clique + [v]
            partition.append(clique)
            
            # Remove edges
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, w = clique[i], clique[j]
                    if w in rem_adj[u]:
                        rem_adj[u].remove(w)
                        rem_adj[w].remove(u)
                        remaining_edges_count -= 1
                        
        # Fallback
        for u in range(n):
            for v in list(rem_adj[u]):
                if u < v:
                    partition.append([u, v])
                    rem_adj[u].remove(v)
                    rem_adj[v].remove(u)
                    
        return partition

    def solve_instance(n, p, seed):
        return greedy_partition(n, p, seed)

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
