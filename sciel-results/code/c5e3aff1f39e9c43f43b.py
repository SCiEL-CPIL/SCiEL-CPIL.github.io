import math
import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 255
    target = 255
    
    # 1. Generate candidate pool: numbers in [1, 255] that share a factor with 255
    # Factors of 255 are 3, 5, 17
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, target) > 1:
            candidates.append(i)
            
    # Ensure 255 is in the list (it should be, gcd(255, 255)=255)
    if target not in candidates:
        candidates.append(target)
        
    # Map values to indices for graph representation
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    m = len(candidates)
    
    # 2. Build adjacency list for the "non-coprime" graph
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Bron-Kerbosch with pivot to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
            
        # Pivot selection: choose u in P union X that maximizes |N(u) intersect P|
        P_union_X = P.union(X)
        if not P_union_X:
            return
            
        max_count = -1
        pivot = None
        for u in P_union_X:
            count = len(P.intersection(adj[u]))
            if count > max_count:
                max_count = count
                pivot = u
                
        # Recurse on P \ N(pivot)
        # We iterate over candidates in P that are NOT neighbors of pivot
        # If P is empty, we already returned.
        # N(pivot) is adj[pivot]
        candidates_to_recurse = P.difference(adj[pivot])
        
        for v in list(candidates_to_recurse):
            bron_kerbosch(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)
            
            if time.time() > deadline:
                return

    # Initialize
    R = set()
    P = set(range(m))
    X = set()
    
    bron_kerbosch(R, P, X)
    
    # If no clique found (should not happen), fallback to multiples of 3
    if not best_clique:
        return [i for i in candidates if i % 3 == 0]
        
    # Convert indices back to values
    result = [candidates[i] for i in best_clique]
    
    # Verify result contains 255 and is valid
    if target not in result:
        # This shouldn't happen if 255 is connected to everything in the max clique, 
        # but if the max clique doesn't contain 255, we need to force it.
        # However, 255 is connected to all candidates (by definition of candidates).
        # So any clique in the candidate graph can be extended to include 255?
        # No, 255 is a specific node. The clique must be a subset of the graph.
        # If the max clique found doesn't include 255, we can't just add it if it's not connected to all others.
        # But 255 IS connected to all candidates.
        # So if C is a clique, C U {255} is a clique iff 255 is connected to all in C.
        # Since all in C are candidates, 255 is connected to all.
        # So the max clique in the graph of candidates will always be extendable to include 255?
        # Wait. The graph vertices ARE the candidates. 255 is one of them.
        # So any clique found is a set of candidates.
        # Does it contain 255? Not necessarily.
        # But we can always add 255 to any clique of candidates because 255 is adjacent to all candidates.
        # So the max clique size in the graph is K.
        # The max clique size containing 255 is K (if 255 is in some max clique) or K+1?
        # No. If C is a max clique of size K, and 255 notin C, then C U {255} is a clique of size K+1.
        # This contradicts C being a max clique.
        # Therefore, every max clique MUST contain 255.
        # Proof: Let C be a max clique. If 255 notin C, then C U {255} is a larger clique (since 255 adj to all in C).
        # Thus, any max clique contains 255.
        
        pass

    return result

# EVOLVE-BLOCK-END
