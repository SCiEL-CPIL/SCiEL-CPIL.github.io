"""Tensor decomposition for 2x3 times 3x3 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indices:
    # U: row i*m + j (i in 0..n-1, j in 0..m-1)
    # V: row j*p + k (j in 0..m-1, k in 0..p-1)
    # W: row k*n + i (k in 0..p-1, i in 0..n-1)
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k.
    # We can decompose this as:
    # T = e_0 ⊗ 1_m ⊗ 1_p + e_1 ⊗ 1_m ⊗ 1_p
    # where e_i is the i-th standard basis vector of size n,
    #       1_m is the all-ones vector of size m,
    #       1_p is the all-ones vector of size p.
    # This gives rank = n = 2.
    
    rank = n  # rank = 2
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each r in 0..rank-1:
    #   U[i*m+j, r] = 1 if i == r else 0
    #   V[j*p+k, r] = 1 for all j, k
    #   W[k*n+i, r] = 1 for all k, i
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                if i == r:
                    U[i * m + j][r] = 1
        
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
        
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    # Verify the decomposition (optional, for debugging)
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    # Fallback to schoolbook if verification fails
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][column] = 1
                                V[j2 * p + k2][column] = 1
                                W[k2 * n + i2][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
