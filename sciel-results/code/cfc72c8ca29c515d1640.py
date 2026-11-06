import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 512
    
    # Precompute LCM compatibility using bitsets
    # adj[i] is a bitmask where bit j is set if lcm(i+1, j+1) <= N
    # Using 0-indexed: index i represents number i+1
    adj = [0] * N
    
    # Precompute for all pairs
    for i in range(1, N + 1):
        ui = i - 1
        for j in range(i, N + 1):
            uj = j - 1
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                adj[ui] |= (1 << uj)
                adj[uj] |= (1 << ui)
    
    # Track best clique found
    best_clique_mask = 0
    best_size = 0
    
    def update_best(mask, size):
        nonlocal best_clique_mask, best_size
        if size > best_size:
            best_clique_mask = mask
            best_size = size
            
    # Greedy initialization to get a good lower bound
    # Order vertices by degree descending
    order = list(range(N))
    order.sort(key=lambda x: adj[x].bit_count(), reverse=True)
    
    greedy_mask = 0
    for v in order:
        if (adj[v] & greedy_mask) == greedy_mask:
            greedy_mask |= (1 << v)
            
    update_best(greedy_mask, greedy_mask.bit_count())
    
    # Bron-Kerbosch with Pivoting
    # P: bitmask of candidate vertices
    # R: bitmask of vertices in current clique
    # X: bitmask of excluded vertices
    
    def bron_kerbosch(P, R, X):
        nonlocal best_clique_mask, best_size
        
        if time.time() > deadline:
            return
        
        # If P is empty, we have a maximal clique
        if P == 0:
            if R.bit_count() > best_size:
                best_clique_mask = R
                best_size = R.bit_count()
            return
        
        # Upper bound pruning: if |R| + |P| <= best_size, no need to continue
        if R.bit_count() + P.bit_count() <= best_size:
            return
        
        # Choose pivot u from P | X that maximizes |P & adj[u]|
        union = P | X
        best_u = -1
        best_count = -1
        temp = union
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            count = (P & adj[u]).bit_count()
            if count > best_count:
                best_count = count
                best_u = u
            temp ^= low
        
        if best_u == -1:
            return
        
        # Candidates: P \ N(best_u)
        candidates = P & ~adj[best_u]
        
        while candidates:
            low = candidates & (-candidates)
            v = low.bit_length() - 1
            candidates ^= low
            
            # Recurse: R + v, P ∩ N(v), X ∩ N(v)
            bron_kerbosch(P & adj[v], R | low, X & adj[v])
            
            # Remove v from P, add to X
            P &= ~low
            X |= low
            
            if time.time() > deadline:
                return
    
    # Run Bron-Kerbosch starting with all vertices
    all_mask = (1 << N) - 1
    bron_kerbosch(all_mask, 0, 0)
    
    # Convert mask to list of 1-indexed values
    result = []
    for i in range(N):
        if (best_clique_mask >> i) & 1:
            result.append(i + 1)
            
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result
