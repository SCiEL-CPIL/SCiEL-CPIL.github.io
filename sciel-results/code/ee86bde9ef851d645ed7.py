import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Finds the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Generate candidate vertices
    # Vertices must share a factor with 255 (3, 5, 17)
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    
    # 2. Build adjacency matrix
    # adj[i][j] is True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True

    # 3. Bron-Kerbosch with pivot
    # We want the maximum clique.
    # Since 255 is in candidates, let's find its index.
    # 255 is connected to all other candidates.
    # So Max Clique of G = {255} U Max Clique of G[neighbors(255)]
    # Neighbors of 255 are all other candidates.
    # So we can just find Max Clique of G and it will include 255?
    # Yes, because 255 is a universal vertex.
    
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
            
        # Choose pivot u from P U X to minimize branching
        # Heuristic: pick u with max degree in P
        if P or X:
            U = list(P) + list(X)
            pivot = max(U, key=lambda u: sum(1 for v in P if adj[u][v]))
        else:
            pivot = None
            
        # Branch on vertices in P that are NOT neighbors of pivot
        # If P is empty, loop doesn't run.
        if pivot is not None:
            candidates_for_branch = [v for v in list(P) if not adj[pivot][v]]
        else:
            candidates_for_branch = list(P)
            
        for v in candidates_for_branch:
            new_R = R | {v}
            new_P = P & {u for u in P if adj[v][u]}
            new_X = X & {u for u in X if adj[v][u]}
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
            
            if time.time() > deadline:
                return

    # Initial call
    # R = empty, P = all candidates, X = empty
    bron_kerbosch(set(), set(range(n)), set())
    
    # Map indices back to values
    if best_clique:
        result = [candidates[i] for i in best_clique]
        # Ensure 255 is included (it should be, but just in case)
        if 255 not in result:
             result.append(255)
        return sorted(result)
    else:
        # Fallback: multiples of 3
        return list(range(3, 256, 3))

# EVOLVE-BLOCK-END
