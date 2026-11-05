"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # We need a subset of [1, 255] containing 255, with gcd(a,b) > 1 for all distinct pairs.
    # Maximize cardinality.
    
    # Candidates: numbers in [1, 255] that share a factor with 255.
    # These are multiples of 3, 5, or 17.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency list using bitmasks for efficiency.
    # adj[i] is a bitmask where bit j is 1 if candidates[i] and candidates[j] have gcd > 1.
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= 1 << j
                adj[j] |= 1 << i

    # The maximum clique containing 255 is 1 (for 255 itself) + maximum clique in N(255).
    # N(255) is the set of candidates connected to 255.
    # Since 255 is divisible by 3, 5, 17, it is connected to all candidates.
    # So we are looking for the maximum clique in the induced subgraph of all candidates.
    # Wait, 255 must be in the set. So we restrict the search to cliques containing 255.
    # This is equivalent to finding the maximum clique in the subgraph induced by N(255) and adding 255.
    
    P0 = adj[idx_255]  # Bitmask of neighbors of 255
    
    best_size = 1
    best_clique = {idx_255}
    
    def popcount(x):
        return bin(x).count('1')
        
    def search(P, size, clique):
        nonlocal best_size, best_clique
        if time.time() > deadline:
            return
        if P == 0:
            if size > best_size:
                best_size = size
                best_clique = set(clique)
            return
        if size + popcount(P) <= best_size:
            return
        
        # Get vertices in P
        P_list = [i for i in range(n) if (P >> i) & 1]
        
        # Greedy coloring for upper bound (Tomita's algorithm)
        # We color the subgraph induced by P.
        # The number of colors is an upper bound on the clique size.
        color_classes = []
        colors = {}
        for v in P_list:
            placed = False
            for ci, cc in enumerate(color_classes):
                # Check if v is independent of all vertices in color class ci
                if all(not (adj[v] >> u) & 1 for u in cc):
                    cc.append(v)
                    colors[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[v] = len(color_classes) - 1
        
        if size + len(color_classes) <= best_size:
            return
        
        # Process vertices in decreasing color order
        P_list.sort(key=lambda v: -colors[v])
        
        for v in P_list:
            if not (P >> v) & 1:
                continue
            if time.time() > deadline:
                return
            new_P = P & adj[v]
            new_size = size + 1
            if new_size + popcount(new_P) <= best_size:
                continue
            search(new_P, new_size, clique | {v})
            P &= ~(1 << v)
            
    # Start search with 255 included
    search(P0, 1, {idx_255})
    
    # If search found a clique, use it
    if best_size > 0:
        result = [candidates[i] for i in best_clique]
        # Ensure 255 is in the result (it should be, as we started with it)
        if 255 not in result:
            result.append(255)
        return sorted(result)
    
    # Fallback: Multiples of 3
    return sorted(list(range(3, 256, 3)))
# EVOLVE-BLOCK-END
