import math
import time

def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 255
    # Candidates must share a factor with 255 (3, 5, 17)
    # 255 = 3 * 5 * 17
    candidates = []
    for x in range(1, n + 1):
        if x % 3 == 0 or x % 5 == 0 or x % 17 == 0:
            candidates.append(x)
            
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
        
    # We must include 255.
    # All other elements must be compatible with 255.
    # Since 255 is divisible by 3, 5, 17, any candidate is compatible with 255.
    # So we find max clique in candidates \ {255}, then add 255.
    
    if 255 in candidates:
        candidates.remove(255)
        
    N = len(candidates)
    if N == 0:
        return [255]
        
    # Map candidates to indices
    # Precompute adjacency bitmasks
    adj = [0] * N
    for i in range(N):
        ci = candidates[i]
        mask = 0
        for j in range(N):
            if i != j:
                cj = candidates[j]
                if math.gcd(ci, cj) > 1:
                    mask |= (1 << j)
        adj[i] = mask
        
    # Bron-Kerbosch with pivot
    # R: current clique (list of indices)
    # P: candidates to add (bitmask)
    # X: candidates already processed (bitmask)
    
    best_clique = []
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        # Pruning: if even adding all of P doesn't beat best, return
        if len(R) + popcount(P) <= best_size:
            return
            
        if P == 0 and X == 0:
            # Found a maximal clique
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
            
        # Pivot selection
        # Choose u from P union X to maximize |P & adj[u]|
        union = P | X
        if union == 0:
            return
            
        pivot = -1
        max_neighbors = -1
        temp = union
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            neigh_count = popcount(P & adj[idx])
            if neigh_count > max_neighbors:
                max_neighbors = neigh_count
                pivot = idx
            temp ^= lsb
            
        # Candidates to try: P minus neighbors of pivot
        if pivot == -1:
            candidates_to_try = P
        else:
            candidates_to_try = P & ~adj[pivot]
            
        temp = candidates_to_try
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            
            # Recurse
            bron_kerbosch(R + [v], P & adj[v], X & adj[v])
            
            # Remove v from P, add to X
            P ^= lsb
            X |= lsb
            temp ^= lsb
            
            if time.time() > deadline:
                return

    # Initial call
    # P = all nodes, X = 0, R = []
    bron_kerbosch([], (1 << N) - 1, 0)
    
    # Extract best clique
    result = [255]
    for i in best_clique:
        result.append(candidates[i])
        
    return result
