import math
import time
import sys

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Build candidate set: all x in [1,255] with gcd(x,255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # Build adjacency using bitsets for speed
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Track best clique found so far (must include 255)
    idx_255 = candidates.index(255)
    
    # Initialize with a known large clique: all multiples of 3
    # This ensures we have a strong baseline and prune more effectively
    init_clique_indices = set()
    for i, val in enumerate(candidates):
        if val % 3 == 0:
            init_clique_indices.add(i)
            
    # Verify 255 is in it (it is, 255 % 3 == 0)
    # Verify it's a clique (it is, all share factor 3)
    
    best_clique = init_clique_indices
    best_size = len(best_clique)
    
    # Bron-Kerbosch with pivot, tracking best globally
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = set(R)
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        U = P | X
        if not U:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = set(R)
            return
            
        # Pick the vertex with maximum degree in P (standard BK pivot)
        pivot = max(U, key=lambda u: len(adj[u] & P))
        
        # For each v in P \ N(pivot)
        candidates_for_recursion = list(P - adj[pivot])
        
        # Sort by degree for better pruning
        candidates_for_recursion.sort(key=lambda v: len(adj[v] & P), reverse=True)
        
        for v in candidates_for_recursion:
            if time.time() > deadline:
                return
            
            # Upper bound: |R| + |P|
            if len(R) + len(P) <= best_size:
                return
            
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
    
    # Start with 255 forced in the clique
    R = {idx_255}
    P = set(adj[idx_255])
    X = set()
    
    bron_kerbosch(R, P, X)
    
    # Convert back to values
    result_set = set(candidates[i] for i in best_clique)
    
    # Ensure 255 is in the result
    if 255 not in result_set:
        result_set.add(255)
    
    # Double-check validity
    res_list = list(result_set)
    if not res_list:
        return [255]
    
    for i in range(len(res_list)):
        for j in range(i+1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                # Fallback to all multiples of 3
                return list(range(3, 256, 3))
    
    if 255 not in res_list:
        return list(range(3, 256, 3))
    
    return res_list

# For quick testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
    print(f"Valid: {valid}")
