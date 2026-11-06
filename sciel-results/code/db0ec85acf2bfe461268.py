import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Identify candidates
    # Must contain 255.
    # Any x in S must have gcd(x, 255) > 1.
    # 255 = 3 * 5 * 17
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    # Map values to indices for faster access
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    n = len(candidates)
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Maximum Clique using Bron-Kerbosch with Pivoting
    # We want the maximum clique.
    # Since the graph is not huge, we can run this.
    
    best_clique = []
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        # Check time limit
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R.copy()
            return
        
        # If P is empty but X is not, this branch cannot lead to a better solution
        # than what is already in X? No, standard BK explores all maximal cliques.
        # To find the maximum, we can prune if len(R) + len(P) <= best_size.
        if len(R) + len(P) <= best_size:
            return
            
        # Choose pivot u from P union X to minimize the number of recursive calls
        # u should be the vertex with the maximum degree in the subgraph induced by P union X
        P_union_X = P.union(X)
        if P_union_X:
            # Find u with max len(adj[u] & P_union_X)
            max_deg = -1
            u = -1
            for v in P_union_X:
                deg = len(adj[v].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    u = v
            # We iterate over P \ N(u)
            candidates_to_iterate = P.difference(adj[u])
        else:
            candidates_to_iterate = P
            
        for v in list(candidates_to_iterate):
            # Recurse
            bron_kerbosch(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
            
            # Remove v from P, add to X
            P.discard(v)
            X.add(v)
            
            if time.time() > deadline:
                return

    # Initial call: R is empty, P is all candidates, X is empty
    # Note: We must ensure 255 is in the final set.
    # Since 255 is connected to all candidates, any maximal clique will either include 255 or be disjoint from it?
    # No, 255 is connected to ALL candidates. So 255 is a universal vertex in this candidate graph.
    # Therefore, the maximum clique MUST contain 255.
    # We can just find the max clique in the whole graph.
    
    initial_P = set(range(n))
    initial_X = set()
    bron_kerbosch(set(), initial_P, initial_X)
    
    # Convert indices back to values
    if best_clique:
        result = [candidates[i] for i in best_clique]
    else:
        # Fallback to multiples of 3 if something went wrong (shouldn't happen)
        result = [i for i in range(1, 256) if i % 3 == 0]
        
    # Ensure 255 is in the result (it should be)
    if 255 not in result:
        # If not, add it? No, that might break gcd conditions if we didn't account for it.
        # But 255 is universal, so it's always in the max clique.
        pass
        
    return result
