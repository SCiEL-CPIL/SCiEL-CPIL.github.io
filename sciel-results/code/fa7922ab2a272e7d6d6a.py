"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-5 decomposition for 3x5x1 slice, replicated for each of p=5 columns.
    # This exploits the structure where for each output column k, the tensor slice
    # T_{i,j,k} is 1 for all i,j. We can write this as sum over r of u_{i,j}^r v_j^r w_k^r.
    # Actually, simpler: for each output column k, we need to decompose the 3x5x1 tensor.
    # The tensor is all-ones 3x5. Rank of all-ones matrix is 1.
    # So for each k, rank is 1. Total rank = p * 1 = 5.
    # Wait, let's check.
    # T_{i,j,k} = 1.
    # We want sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1.
    # Let rank = p = 5.
    # Let r index k.
    # For a given r (which corresponds to output column k=r):
    # U_{i*m+j, r} should be 1 if j is "active" for this decomposition step?
    # V_{j*p+k, r} should be 1 if j=k? No.
    # Let's try:
    # U_{i*m+j, r} = 1 for all i,j? No, U has shape (n*m, rank).
    # Let's use the fact that sum_i is not involved in the tensor value, but i is an index.
    # The tensor is T[i, j, k] = 1.
    # This is the outer product of three all-ones vectors: (1,...,1) (1,...,1) (1,...,1).
    # Rank of a tensor which is an outer product of three vectors is 1.
    # So the whole 3x5x5 tensor has rank 1?
    # Let's check.
    # T[i,j,k] = u_i v_j w_k. If u_i=1, v_j=1, w_k=1, then T[i,j,k]=1.
    # So rank 1 should work?
    # Let's verify with the interface.
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Sum over r (only r=0): U[i*m+j,0] V[j*p+k,0] W[k*n+i,0] = 1*1*1 = 1.
    # This matches T[i,j,k]=1.
    # So rank 1 is sufficient!
    
    rank = 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
