import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 110  # 10s margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        cliques = []
        
        while True:
            if time.time() > deadline:
                break
                
            # Find the edge (u, v) with the maximum common neighbors in the uncovered graph
            best_u = -1
            best_v = -1
            best_count = -1
            
            # Iterate over all pairs to find the "best" seed edge
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        # Edge (u,v) exists in uncovered graph
                        common = adj[u] & adj[v]
                        # Count common neighbors
                        count = bin(common).count('1')
                        if count > best_count:
                            best_count = count
                            best_u = u
                            best_v = v
                            
            if best_u == -1:
                # No edges left
                break
                
            # Build clique starting with best_u, best_v
            clique = [best_u, best_v]
            
            # Candidates are common neighbors
            candidates_mask = adj[best_u] & adj[best_v]
            
            # Find a maximal clique in the subgraph induced by candidates_mask
            # using greedy approach
            # We want to add vertices to clique such that they are connected to all existing clique members.
            # Since all candidates are connected to best_u and best_v, we just need them to be connected to each other
            # and to any previously added candidates.
            
            # Greedy Maximal Clique on induced subgraph of candidates_mask
            # We maintain a mask of valid candidates
            valid_mask = candidates_mask
            
            while valid_mask:
                # Pick vertex with highest degree in the induced subgraph
                # Degree of w in induced subgraph = popcount(adj[w] & valid_mask)
                best_w = -1
                best_deg = -1
                m = valid_mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    deg = bin(adj[w] & valid_mask).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
                    m ^= lsb
                
                if best_w == -1:
                    break
                    
                # Add best_w to clique
                clique.append(best_w)
                
                # Remove best_w and its non-neighbors from valid_mask
                # valid_mask becomes vertices in valid_mask that are neighbors of best_w
                valid_mask &= adj[best_w]
                
            cliques.append(clique)
            
            # Remove edges in this clique from adj
            for i in range(len(clique)):
                ci = clique[i]
                for j in range(i + 1, len(clique)):
                    cj = clique[j]
                    adj[ci] &= ~(1 << cj)
                    adj[cj] &= ~(1 << ci)
                    
        return cliques

    c1 = solve(32, 0.6, 601)
    c2 = solve(64, 0.55, 602)
    
    return [c1, c2]
