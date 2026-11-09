"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85

    # Build candidate set: numbers sharing a factor with 255 = 3*5*17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    
    # Precompute adjacency as sets for fast membership
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    idx_255 = candidates.index(255)
    
    # Start with the known large clique: all multiples of 3 (size 85)
    best_set = set(i for i in range(n) if candidates[i] % 3 == 0)
    best_size = len(best_set)
    
    # Local search: try to improve by remove-one-add-many
    improved = True
    while improved and (time.time() - start_time < deadline - 5):
        improved = False
        S = set(best_set)
        
        for rem in list(S):
            if rem == idx_255:
                continue
            S_t = S - {rem}
            # Add all compatible elements
            for add in range(n):
                if add in S_t:
                    continue
                if all(add in adj[y] or add == y for y in S_t):
                    S_t.add(add)
            if len(S_t) > len(S):
                best_set = S_t
                best_size = len(S_t)
                improved = True
                break
    
    # Try starting from other large cliques
    if time.time() - start_time < deadline - 10:
        for prime in [5, 17]:
            if time.time() - start_time > deadline - 5:
                break
            S = set(i for i in range(n) if candidates[i] % prime == 0)
            if idx_255 not in S:
                continue
            # Expand
            changed = True
            while changed and (time.time() - start_time < deadline - 5):
                changed = False
                for add in range(n):
                    if add in S:
                        continue
                    if all(add in adj[y] or add == y for y in S):
                        S.add(add)
                        changed = True
                        break
            # Remove-one-add-many
            imp2 = True
            while imp2 and (time.time() - start_time < deadline - 5):
                imp2 = False
                for rem in list(S):
                    if rem == idx_255:
                        continue
                    S_t = S - {rem}
                    for add in range(n):
                        if add in S_t:
                            continue
                        if all(add in adj[y] or add == y for y in S_t):
                            S_t.add(add)
                    if len(S_t) > len(S):
                        S = S_t
                        imp2 = True
                        break
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
    
    return sorted(candidates[i] for i in best_set)

# EVOLVE-BLOCK-END
