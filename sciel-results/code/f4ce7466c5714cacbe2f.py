import time
import sys
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    adj = [[False] * n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = True
                adj[v][u] = True
                edges.append((u, v))
    return adj, edges

def _find_maximal_cliques(adj, n, min_size):
    """
    Finds all maximal cliques of size >= min_size using Bron-Kerbosch with pivot.
    """
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(list(R))
            return
            
        # Choose pivot u from P U X
        # We want to minimize iterations, so pick u with max degree in P
        pivot = -1
        max_deg = -1
        for u in P:
            deg = 0
            for v in P:
                if adj[u][v]:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                pivot = u
        
        if pivot == -1:
            # P is empty, handled above? No, if P is empty, loop doesn't run.
            # If P is not empty, we need to pick a pivot.
            # If P is empty, bronk returns if X is empty.
            # If P is not empty, we iterate over P \ N(pivot)
            pass
            
        # Iterate over P \ N(pivot)
        # If pivot is -1 (P empty), we don't enter this loop, but we handled base case.
        # Wait, if P is not empty, we must have a pivot.
        # If P is empty, the function returns at the start.
        
        # Create list of candidates
        candidates = []
        for u in P:
            if pivot == -1 or not adj[u][pivot]:
                candidates.append(u)
                
        for u in candidates:
            new_R = R | {u}
            new_P = P & {v for v in P if adj[u][v]}
            new_X = X & {v for v in X if adj[u][v]}
            bronk(new_R, new_P, new_X)
            P = P - {u}
            X = X | {u}

    # Initialize
    R = set()
    P = set(range(n))
    X = set()
    bronk(R, P, X)
    return cliques

def _find_triangles(adj, n):
    """
    Finds all triangles in the graph.
    """
    triangles = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                # Find common neighbors w > v to avoid duplicates
                # Actually, just check w > v
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        triangles.add((u, v, w))
    return list(triangles)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback to baseline if time runs out
            rng = Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            results.append([list(e) for e in edges])
            continue
            
        adj, edges = _random_graph(n, p, seed)
        edge_set = set(edges)
        covered_edges = set()
        partition = []
        
        # Step 1: Maximal cliques of size >= 4
        # For n=64, this might be slow if not careful.
        # Let's limit the search or use a heuristic if it's too slow.
        # Bron-Kerbosch on dense graphs can be exponential in worst case.
        # However, for random graphs, it's usually fine.
        
        # Optimization: Only run Bron-Kerbosch if n is small enough or time permits.
        # For n=32, it's fast. For n=64, it might be slower.
        # Let's try.
        
        large_cliques = []
        if time.time() < deadline - 20.0:
            large_cliques = _find_maximal_cliques(adj, n, 4)
            
        # Sort by size descending
        large_cliques.sort(key=len, reverse=True)
        
        for clique in large_cliques:
            if time.time() > deadline - 10.0:
                break
            # Check if all edges in clique are uncovered
            valid = True
            clique_edges = []
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u, v), max(u, v))
                    if e in covered_edges:
                        valid = False
                        break
                    clique_edges.append(e)
                if not valid:
                    break
            
            if valid:
                partition.append(clique)
                covered_edges.update(clique_edges)
                
        # Step 2: Triangles
        if time.time() < deadline - 10.0:
            triangles = _find_triangles(adj, n)
            # Sort triangles? Maybe by vertex degrees?
            # For now, just iterate.
            # To be greedy, we should pick those that are "most constrained" or just first available.
            # Since they are all size 3, order doesn't change count, only which ones are picked.
            # Any maximal set of edge-disjoint triangles is fine for count?
            # Yes, each saves 2.
            
            for tri in triangles:
                if time.time() > deadline - 5.0:
                    break
                u, v, w = tri
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, w), max(v, w))
                e3 = (min(u, w), max(u, w))
                
                if e1 not in covered_edges and e2 not in covered_edges and e3 not in covered_edges:
                    partition.append([u, v, w])
                    covered_edges.update([e1, e2, e3])
                    
        # Step 3: Remaining edges as K2s
        for u, v in edges:
            e = (min(u, v), max(u, v))
            if e not in covered_edges:
                partition.append([u, v])
                
        results.append(partition)
        
    return results
