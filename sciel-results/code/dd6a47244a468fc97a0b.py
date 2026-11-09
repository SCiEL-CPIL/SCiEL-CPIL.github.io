"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for O(1) lookup
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    # Compute degree for ordering
    degrees = [len(adj[i]) for i in range(n)]
    
    # Bron-Kerbosch with pivot to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Choose pivot u from P union X to maximize branching
        u = None
        max_neighbors = -1
        candidates_for_pivot = list(P) + list(X)
        for v in candidates_for_pivot:
            count = len(adj_set[v] & P)
            if count > max_neighbors:
                max_neighbors = count
                u = v
        
        if u is None:
            candidates = list(P)
        else:
            candidates = [v for v in P if v not in adj_set[u]]
        
        # Sort candidates by degree (higher degree first) for better pruning
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        
        for v in candidates:
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P & adj_set[v]
            new_X = X & adj_set[v]
            bron_kerbosch(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    # Run BK with different orderings
    # First: natural order
    bron_kerbosch(set(), set(range(n)), set())
    
    # Second: by degree (descending)
    if time.time() < deadline:
        order_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        # We can't easily change the order in BK, but we can try multiple random restarts
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            random.shuffle(order_by_degree)
            # Try BK with a subset or different approach
            # Actually, let's just run BK again - it's deterministic, so same result
            # Instead, let's try a greedy approach to find a good clique
            pass
    
    # Greedy local search to potentially improve
    def greedy_clique():
        """Find a clique using greedy approach with randomization."""
        remaining = set(range(n))
        clique = set()
        while remaining:
            # Pick vertex with max degree in remaining subgraph
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(adj_set[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None or best_deg < 0:
                break
            clique.add(best_v)
            remaining = remaining & adj_set[best_v]
        return clique
    
    # Try multiple greedy starts
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        # Randomize by shuffling and picking
        shuffled = list(range(n))
        random.shuffle(shuffled)
        remaining = set(shuffled)
        clique = set()
        while remaining:
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(adj_set[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None or best_deg < 0:
                break
            clique.add(best_v)
            remaining = remaining & adj_set[best_v]
        
        if len(clique) > len(best_clique):
            best_clique = list(clique)
    
    # Local search: try to add vertices to existing clique
    def local_search_improve(clique):
        """Try to improve a clique by swapping vertices."""
        clique_set = set(clique)
        improved = True
        while improved:
            improved = False
            # Try adding each non-clique vertex
            for v in range(n):
                if v in clique_set:
                    continue
                # Check if v can be added
                if all(v in adj_set[u] for u in clique_set):
                    clique_set.add(v)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one clique vertex, add two others
            for u in list(clique_set):
                new_clique = clique_set - {u}
                # Find two vertices that can both be added
                candidates = [v for v in range(n) if v not in new_clique and all(v in adj_set[x] for x in new_clique)]
                # Try pairs
                for i in range(len(candidates)):
                    for j in range(i+1, len(candidates)):
                        v1, v2 = candidates[i], candidates[j]
                        if v1 in adj_set[v2]:
                            # Check if adding both improves
                            if len(new_clique) + 2 > len(clique_set):
                                clique_set = new_clique | {v1, v2}
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return clique_set
    
    # Apply local search to best clique
    if best_clique:
        improved = local_search_improve(best_clique)
        if len(improved) > len(best_clique):
            best_clique = list(improved)
    
    # Fallback if no clique found
    if not best_clique:
        if edges:
            u, v = edges[0]
            best_clique = [u, v]
        else:
            best_clique = [0, 1]

    # Construct result
    branches = best_clique
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u = branches[i]
            v = branches[j]
            paths.append([u, v])
            
    # Ensure types are correct
    return {
        'branches': list(branches),
        'paths': [list(p) for p in paths]
    }

# EVOLVE-BLOCK-END
