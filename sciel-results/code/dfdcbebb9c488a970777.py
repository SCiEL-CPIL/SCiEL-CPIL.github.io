import time
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin: 60s limit, use 55s
    deadline = start_time + 55.0
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # 1. Build adjacency bitsets
    # adj[i] is an integer where bit j is set if lcm(i, j+1) <= 512
    # We use 0-indexed bits for nodes 1..512. Bit (j-1) corresponds to node j.
    adj = [0] * (n + 1)
    
    # Precompute gcds or just compute on the fly?
    # 512*512/2 = ~131k pairs. gcd is fast.
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                # Set bit j-1 in adj[i] and bit i-1 in adj[j]
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
        
        if time.time() > deadline:
            return [1, 2]

    # 2. Generate initial best clique: Divisors of L with max divisors
    best_seed = []
    max_divs = 0
    for L in range(1, n + 1):
        divs = []
        for k in range(1, int(L**0.5) + 1):
            if L % k == 0:
                divs.append(k)
                if k != L // k:
                    divs.append(L // k)
        if len(divs) > max_divs:
            max_divs = len(divs)
            best_seed = divs
            
    best_clique_size = len(best_seed)
    best_clique = best_seed
    
    # 3. Bron-Kerbosch with bitsets and Pivot
    # R: list of nodes in current clique
    # P: bitset of candidates
    # X: bitset of excluded nodes
    
    # To speed up time checks, we use a simple counter or less frequent checks
    # But for N=512, time.time() is okay if not in innermost tight loop of bit ops?
    # It is in the recursive call.
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_clique_size
        
        # Pruning: Even if we add all remaining candidates, we can't beat best
        if len(R) + P.bit_count() <= best_clique_size:
            return

        if P == 0:
            if X == 0:
                if len(R) > best_clique_size:
                    best_clique_size = len(R)
                    best_clique = list(R)
            return

        # Pivot selection: u in P | X that maximizes (P & adj[u]).bit_count()
        # We only look in P for simplicity, which is standard for "Tomita" style
        # If P is empty, we returned.
        
        pivot = -1
        max_count = -1
        
        p_temp = P
        # Optimization: If P has very few bits, iterate them.
        # If P has many bits, maybe sample? No, 512 is small.
        while p_temp:
            lsb = p_temp & -p_temp
            u = lsb.bit_length()
            
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            p_temp ^= lsb
            
        if pivot == -1:
            return

        # Candidates: P \ N(pivot)
        # In our adj representation, adj[pivot] contains neighbors.
        # We want nodes in P that are NOT neighbors of pivot.
        # Note: pivot itself is not a neighbor of itself in our adj construction (only i!=j).
        # So if pivot is in P, it is in candidates.
        candidates = P & ~adj[pivot]
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length()
            
            # Recurse
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            R.append(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()
            
            # Update P and X
            P ^= lsb # Remove v from P
            X |= lsb # Add v to X
            
            candidates ^= lsb # Remove v from candidates
            
            # Time check
            if time.time() > deadline:
                return

    # Initial call
    # P = all nodes
    initial_P = (1 << n) - 1
    initial_X = 0
    initial_R = []
    
    # To improve chances, we can start with the best_seed as R?
    # No, standard BK starts empty. But the pruning uses best_clique_size.
    # If we start with R=best_seed, P = neighbors of all in best_seed, X=0.
    # This might be faster as it starts deeper.
    # Let's try starting with the seed.
    
    # Calculate P for the seed
    # P must be a subset of nodes that are compatible with ALL nodes in seed.
    # P = (1<<n)-1
    # for v in best_seed: P &= adj[v]
    
    seed_P = (1 << n) - 1
    for v in best_seed:
        seed_P &= adj[v]
        
    # Remove seed nodes from P
    for v in best_seed:
        seed_P &= ~(1 << (v - 1))
        
    # X should be nodes that are NOT compatible with some seed node?
    # Actually, if we fix R=seed, we are looking for extensions.
    # Any node not in seed and not in P is "excluded" from being added to THIS branch.
    # But standard BK logic with R=seed, P=compatible_others, X=0 is valid for finding cliques containing seed.
    # However, there might be a larger clique NOT containing the seed.
    # So we must also run the standard search from empty R.
    # Given the strong seed, the standard search will prune heavily.
    
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    return best_clique
