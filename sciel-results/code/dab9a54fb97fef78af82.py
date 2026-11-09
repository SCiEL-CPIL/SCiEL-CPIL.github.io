"""Program for large pairwise noncoprime sets containing the interval endpoint."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy:
    1. Identify candidates: numbers in [1, 255] with gcd(x, 255) > 1.
    2. Build a graph where an edge exists if gcd(u, v) > 1.
    3. Find the Maximum Clique in this graph that contains 255 using Bron-Kerbosch with pivot.
    """
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Initialize best_clique with multiples of 3 (a known valid clique containing 255)
    best_clique = set()
    for i, val in enumerate(candidates):
        if val % 3 == 0:
            best_clique.add(i)
    
    # Initial state for Bron-Kerbosch
    R = {idx_255}
    P = set(adj[idx_255])
    X = set()
    
    # Bron-Kerbosch with pivot and pruning
    def bk(R, P, X):
        nonlocal best_clique
        
        # Pruning: if even adding all P to R doesn't beat best, return
        if len(R) + len(P) <= len(best_clique):
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
            
        # Pivot selection
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            candidates_to_process = P - adj[u]
        else:
            candidates_to_process = set()
            
        for v in candidates_to_process:
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bk(R, P_new, X_new)
            
            R.remove(v)
            P.remove(v)
            X.add(v)

    bk(R, P, X)
    
    # Convert indices back to values
    result = [candidates[i] for i in best_clique]
    
    return result
# EVOLVE-BLOCK-END
