<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all triples (i, j, k)
    for idx in range(rank):
        # Map linear index to (i, j, k)
        # Order: i * m * p + j * p + k
        k = idx % p
        tmp = idx // p
        j = tmp % m
        i = tmp // m
        
        col = idx
        
        # U index: i * m + j
        u_idx = i * m + j
        U[u_idx][col] = 1
        
        # V index: j * p + k
        v_idx = j * p + k
        V[v_idx][col] = 1
        
        # W index: k * n + i
        w_idx = k * n + i
        W[w_idx][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor is a direct sum of n all-ones matrices of size m x p.
    # Each has rank 1. Total rank is n.
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each i in 0..n-1, assign rank index r=i
    for i in range(n):
        r = i
        # U[i, j, r] = 1 for all j
        for j in range(m):
            u_idx = i * m + j
            U[u_idx][r] = 1
            
        # V[j, k, r] = 1 for all j, k
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                V[v_idx][r] = 1
                
        # W[k*n + i, r] = 1 for all k
        for k in range(p):
            w_idx = k * n + i
            W[w_idx][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
