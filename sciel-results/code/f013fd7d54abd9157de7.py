"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 512] with lcm(a,b) <= 512 for every pair,
    maximizing cardinality. Uses Bron-Kerbosch with pivot to find the exact
    maximum clique in the compatibility graph.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    N = 512
    # Build adjacency list
    # adj[u] is a set of v > u such that lcm(u, v) <= N
    # We store indices 0..N-1 corresponding to numbers 1..N
    adj = [set() for _ in range(N)]
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # lcm(u, v) = u * v // gcd(u, v)
            # Check if lcm <= N
            g = gcd(u, v)
            # To avoid overflow or division issues, check (u // g) * v <= N
            if (u // g) * v <= N:
                adj[u - 1].add(v - 1)
                adj[v - 1].add(u - 1)

    # Bron-Kerbosch with pivot
    best_clique = []
    
    def bronk(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        # u is chosen to minimize the number of recursive calls
        pivot_candidates = P | X
        if not pivot_candidates:
            return
            
        max_degree = -1
        pivot = None
        for u in pivot_candidates:
            deg = len(P & adj[u])
            if deg > max_degree:
                max_degree = deg
                pivot = u
                if deg == len(P):
                    break
        
        # Recurse on P \ N(pivot)
        # The candidates are those in P that are NOT adjacent to pivot
        candidates = P - adj[pivot]
        
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    # Initial call
    all_vertices = set(range(N))
    bronk(set(), all_vertices, set())
    
    # Convert back to 1-based indices
    if not best_clique:
        # Fallback if something went wrong (should not happen)
        return [1]
        
    result = sorted([v + 1 for v in best_clique])
    return result

# EVOLVE-BLOCK-END
