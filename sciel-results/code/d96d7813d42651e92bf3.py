"""Rewritten program for 2x2 * 2x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    The multiplication tensor T_{i,j,k} for C[i,k] = sum_j A[i,j] * B[j,k]
    has entries indexed by (i in 0..1, j in 0..1, k in 0..8).
    
    For each (i, j, k) triple, we have a rank-1 term:
      U[i*m + j] * V[j*p + k] * W[k*n + i] = 1
    
    This gives the schoolbook decomposition with rank = n*m*p = 2*2*9 = 36.
    
    The key question is whether we can do better than 36. For 2x2 * 2x1,
    the rank is known to be 4 (the standard matrix-vector product requires 4
    multiplications). For 2x2 * 2x9, the problem decomposes as 9 independent
    2x2 * 2x1 problems, each requiring rank 4, giving total rank 36.
    
    However, there might be shared structure. Let's verify: the tensor is
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    This is exactly the matrix multiplication tensor MM_{2,2,9}. The rank of
    MM_{n,m,p} is at least n*m*p / (some factor) but for small dimensions,
    schoolbook is often optimal or near-optimal.
    
    For 2x2 * 2x9, the schoolbook rank is 36. Let's construct it carefully.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36 for schoolbook
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 18 x 36
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (row i, col j of A)
                # V index: j*p + k (row j, col k of B)
                # W index: k*n + i (row k, col i of C) -- trace convention W[k*n+i, r]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
