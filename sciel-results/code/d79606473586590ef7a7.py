"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy: Build a compatibility graph on candidates (numbers in [1,255] sharing a factor with 255),
    then find the maximum clique containing 255 using Bron-Kerbosch with pivoting.
    """
    start_time = time.time()
    deadline = 85  # Safety margin before 90s limit
    
    # Build candidate set: numbers in [1,255] that share a factor with 255 = 3*5*17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    # Index 255
    idx_255 = candidates.index(255)
    
    # Build adjacency matrix: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Bron-Kerbosch with pivoting to find maximum clique
    best_clique = [idx_255]  # Must contain 255
    best_size = 1
    
    # For maximum clique containing 255, we can restrict to neighbors of 255
    # All candidates share a factor with 255, so all are neighbors of 255
    
    def bron_kerbosch(R, P, X, depth=0):
        nonlocal best_clique, best_size, deadline
        
        if time.time() - start_time > deadline:
            return
        
        if len(P) == 0 and len(X) == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Pruning: even if we add all of P, can't beat best
        if len(R) + len(P) <= best_size:
            return
        
        # Pivoting: choose vertex in P ∪ X with max degree in P
        P_X = P | X
        if P_X:
            pivot = max(P_X, key=lambda v: len(P & adj_set[v]))
            candidates_to_iter = P - adj_set[pivot]
        else:
            candidates_to_iter = P()
        
        for v in list(candidates_to_iter):
            if time.time() - start_time > deadline:
                return
            new_R = R | {v}
            new_P = P & adj_set[v]
            new_X = X & adj_set[v]
            P.discard(v)
            X.add(v)
            bron_kerbosch(new_R, new_P, new_X, depth + 1)
            P.add(v)
            X.discard(v)
    
    # Convert adj to sets for faster intersection
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_set[i].add(j)
    
    # Initialize with R={255}, P=neighbors of 255 excluding 255, X=empty
    R = {idx_255}
    P = set(adj_set[idx_255]) - {idx_255}
    X = set()
    
    # Since all candidates share a factor with 255, P = all other candidates
    bron_kerbosch(R, P, X)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    return sorted(result)

# EVOLVE-BLOCK-END
