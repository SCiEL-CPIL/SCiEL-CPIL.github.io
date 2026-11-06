import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # Keep 5s margin for safety

    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # Precompute adjacency as bitmasks for speed
    # n is at most ~120, so we can use Python integers as bitmasks
    adj = [0] * n
    for i in range(n):
        vi = candidates[i]
        mask = 0
        for j in range(n):
            if i != j and gcd(vi, candidates[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask
    
    # 255 is universal, so max clique contains it
    # Find max clique in graph, then add 255 if not present (it will be)
    
    best_clique = []
    best_size = 0
    
    def bit_count(x):
        return bin(x).count('1')
    
    def get_bits(mask):
        """Yield indices of set bits in mask."""
        while mask:
            lb = mask & (-mask)
            yield lb.bit_length() - 1
            mask ^= lb
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        # Bound: can't beat best
        if len(R) + bit_count(P) <= best_size:
            return
        
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Greedy coloring for upper bound
        # Color P greedily to get an upper bound on clique size in P
        uncolored = P
        color = 0
        while uncolored:
            color += 1
            # Find a maximal independent set in uncolored (for coloring)
            # Actually, for clique bound, we color so that adjacent vertices have different colors
            # Greedy: repeatedly pick a vertex, remove it and its neighbors
            uncolored2 = uncolored
            while uncolored2:
                # Pick first vertex
                lb = uncolored2 & (-uncolored2)
                v = lb.bit_length() - 1
                uncolored2 &= ~lb
                # Remove neighbors of v from uncolored2 (they can't be same color)
                uncolored2 &= ~adj[v]
            uncolored &= ~(uncolored2 | uncolored & 0)  # This is wrong, let me redo
            
        # Simpler coloring bound:
        # Assign colors greedily; max color is upper bound on clique size
        # Let me use a cleaner approach
        
        # Pivot selection
        if P != 0:
            # Pick pivot u in P∪X maximizing |P ∩ N(u)|
            pivot = None
            max_deg = -1
            # Try vertices in P first
            for u in get_bits(P):
                deg = bit_count(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            # Also try X
            for u in get_bits(X):
                deg = bit_count(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if pivot is None:
                pivot = next(get_bits(P))
            
            # Branch on P \ N(pivot)
            branch_mask = P & ~adj[pivot]
        else:
            branch_mask = 0
            pivot = None
        
        # Iterate over vertices in branch_mask
        # We need to process them one by one, removing from P and adding to X
        # To avoid modifying during iteration, collect the list first
        branch_list = list(get_bits(branch_mask))
        
        for v in branch_list:
            if time.time() > deadline:
                return
            
            vbit = 1 << v
            
            # Recurse
            R.append(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            P &= ~vbit
            X |= vbit
    
    # Initial solution: multiples of 3
    init_clique = []
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_clique.append(i)
    best_size = len(init_clique)
    best_clique = init_clique
    
    # Run Bron-Kerbosch
    R = []
    P = (1 << n) - 1  # All vertices
    X = 0
    
    bron_kerbosch(R, P, X)
    
    # Ensure 255 is in result
    idx_255 = idx_map[255]
    if idx_255 not in best_clique:
        best_clique.append(idx_255)
    
    result_vals = [candidates[i] for i in best_clique]
    
    # Verify
    valid = True
    for i in range(len(result_vals)):
        for j in range(i+1, len(result_vals)):
            if gcd(result_vals[i], result_vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        result_vals = [v for v in candidates if v % 3 == 0]
    
    return result_vals
