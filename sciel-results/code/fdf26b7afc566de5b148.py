"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Build conflict graph: edge (a, b) if (a+b) divides a*b
    # Equivalently, for distinct a, b: (a*b) % (a+b) == 0
    # We want maximum independent set (no two selected vertices are adjacent)
    
    # Precompute conflicts using bitmasks
    # Use Python integers as bitmasks for speed
    # vertex i corresponds to number i+1 (0-indexed)
    
    N = n
    # adj[i] is a bitmask of all vertices j != i such that (i+1, j+1) conflict
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Maximum Independent Set via branch and bound with bitsets
    # Tomita-style algorithm
    import sys
    sys.setrecursionlimit(10000)
    
    best_set = set()
    best_size = 0
    
    def update_best(S, size):
        nonlocal best_set, best_size
        if size > best_size:
            best_size = size
            best_set = set(S)
    
    # P is the set of candidate vertices (as bitmask)
    # S is the current independent set (as list)
    def branch_and_bound(P, S, size):
        nonlocal best_size, best_set
        # Upper bound: size + popcount(P)
        if size + bin(P).count('1') <= best_size:
            return
        
        if P == 0:
            if size > best_size:
                best_size = size
                best_set = set(S)
            return
        
        # Heuristic: pick vertex with maximum degree within P
        # For speed, just pick the lowest set bit or use a simple heuristic
        # Let's pick the vertex with the highest degree in the subgraph induced by P
        best_v = -1
        best_deg = -1
        temp = P
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            # Count neighbors of v within P
            deg = bin(adj[v] & P).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        v = best_v
        lb = 1 << v
        
        # Option 1: include v
        new_S = S + [v + 1]
        new_P = P & ~(adj[v] | lb)  # Remove v and its neighbors
        branch_and_bound(new_P, new_S, size + 1)
        
        # Option 2: exclude v
        branch_and_bound(P & ~lb, S, size)
    
    full_mask = (1 << N) - 1
    branch_and_bound(full_mask, [], 0)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
