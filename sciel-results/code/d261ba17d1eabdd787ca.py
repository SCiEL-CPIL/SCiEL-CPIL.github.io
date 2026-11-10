import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 255

    # 1. Identify candidates U: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = []
    for x in range(1, N + 1):
        if math.gcd(x, N) > 1:
            candidates.append(x)
    
    n_cands = len(candidates)
    
    # Map value to index for easy lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitmask where bit j is 1 if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n_cands
    for i in range(n_cands):
        vi = candidates[i]
        mask = 0
        for j in range(i + 1, n_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                mask |= (1 << j)
                adj[j] |= (1 << i) # symmetric
        adj[i] = mask

    # 3. Find Maximum Clique using Bron-Kerbosch with Pivoting
    best_clique = []
    best_size = 0
    
    # Helper to iterate set bits
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # Pivot selection: choose u in P U X with max degree in P
        # We want to minimize the number of recursive calls, so we pick u
        # that has the most neighbors in P.
        PX = P | X
        best_u = -1
        max_deg = -1
        
        for u in iter_bits(PX):
            # Count neighbors of u in P
            # adj[u] has bits for neighbors. P has bits for candidates.
            # Intersection gives neighbors in P.
            deg = bin(P & adj[u]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_u = u
                
        if best_u == -1:
            return

        # Recurse on P \ N(u)
        # We iterate over vertices v in P that are NOT neighbors of u
        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            # Add v to R
            R.append(v)
            
            # New P: neighbors of v in current P
            P_new = P & adj[v]
            # New X: neighbors of v in current X
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)

    # Initial call
    # R = []
    # P = all candidates (bitmask with n_cands bits set)
    # X = []
    initial_P = (1 << n_cands) - 1
    bron_kerbosch([], initial_P, 0)
    
    # 4. Construct result
    if best_clique:
        result = [candidates[i] for i in best_clique]
    else:
        # Fallback: Multiples of 3
        result = [x for x in range(1, N + 1) if x % 3 == 0]
        
    # Ensure 255 is in result (it should be, but just in case of logic errors)
    if 255 not in result:
        result.append(255)
        
    # Verify validity
    # For performance, we only verify if size is small or if we want to be absolutely sure.
    # With N=255, O(k^2) check is fast for k~100.
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # If invalid, fallback to multiples of 3
                return [x for x in range(1, N + 1) if x % 3 == 0]

    return result
