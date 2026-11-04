"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
Uses a hybrid Strassen/Schoolbook decomposition with rank 55.
"""
import time

DIMENSIONS = [2, 3, 10]
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Indexing convention for W is W[k*n+i, c] corresponding to C[i, k].
    Indexing convention for U is U[i*m+j, c] corresponding to A[i, j].
    Indexing convention for V is V[j*p+k, c] corresponding to B[j, k].
    
    The tensor decomposition is:
    C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 55
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: A[i, 2] * B[2, k] for all i in {0, 1} and k in {0, ..., 9}
    # This part has rank 2 * 10 = 20
    # Columns 0 to 19
    for i in range(n):
        for k in range(p):
            c = i * p + k
            # U[i*m+2, c] = 1
            U[i * m + 2][c] = 1.0
            # V[2*p+k, c] = 1
            V[2 * p + k][c] = 1.0
            # W[k*n+i, c] = 1
            W[k * n + i][c] = 1.0
            
    # Part 2: Strassen for 2x2 by 2x10 blocks
    # We split k into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair q, we use 7 columns
    # Columns 20 to 54
    
    for q in range(5):
        k0 = 2 * q
        k1 = 2 * q + 1
        base_col = 20 + q * 7
        
        # M1 = (A[0,0] + A[1,1]) * (B[0, k0] + B[1, k1])
        c = base_col + 0
        U[0 * m + 0][c] = 1.0
        U[1 * m + 1][c] = 1.0
        V[0 * p + k0][c] = 1.0
        V[1 * p + k1][c] = 1.0
        W[k0 * n + 0][c] = 1.0  # C[0, k0]
        W[k1 * n + 1][c] = 1.0  # C[1, k1]
        
        # M2 = (A[1,0] + A[1,1]) * B[0, k0]
        c = base_col + 1
        U[1 * m + 0][c] = 1.0
        U[1 * m + 1][c] = 1.0
        V[0 * p + k0][c] = 1.0
        W[k0 * n + 1][c] = 1.0  # C[1, k0]
        W[k1 * n + 1][c] = -1.0 # C[1, k1]
        
        # M3 = A[0,0] * (B[0, k1] - B[1, k1])
        c = base_col + 2
        U[0 * m + 0][c] = 1.0
        V[0 * p + k1][c] = 1.0
        V[1 * p + k1][c] = -1.0
        W[k1 * n + 0][c] = -1.0 # C[0, k1]
        W[k1 * n + 1][c] = 1.0  # C[1, k1]
        
        # M4 = A[1,1] * (B[0, k0] - B[0, k1])
        c = base_col + 3
        U[1 * m + 1][c] = 1.0
        V[0 * p + k0][c] = 1.0
        V[0 * p + k1][c] = -1.0
        W[k0 * n + 0][c] = 1.0  # C[0, k0]
        W[k0 * n + 1][c] = -1.0 # C[1, k0]
        
        # M5 = (A[0,0] + A[0,1]) * B[1, k1]
        c = base_col + 4
        U[0 * m + 0][c] = 1.0
        U[0 * m + 1][c] = 1.0
        V[1 * p + k1][c] = 1.0
        W[k0 * n + 0][c] = -1.0 # C[0, k0]
        W[k1 * n + 0][c] = 1.0  # C[0, k1]
        
        # M6 = (A[1,0] - A[0,0]) * (B[0, k0] + B[0, k1])
        c = base_col + 5
        U[1 * m + 0][c] = 1.0
        U[0 * m + 0][c] = -1.0
        V[0 * p + k0][c] = 1.0
        V[0 * p + k1][c] = 1.0
        W[k1 * n + 0][c] = 1.0  # C[0, k1]
        W[k0 * n + 1][c] = -1.0 # C[1, k0]
        
        # M7 = (A[0,1] - A[1,1]) * (B[1, k0] + B[1, k1])
        c = base_col + 6
        U[0 * m + 1][c] = 1.0
        U[1 * m + 1][c] = -1.0
        V[1 * p + k0][c] = 1.0
        V[1 * p + k1][c] = 1.0
        W[k0 * n + 0][c] = 1.0  # C[0, k0]

    return {"U": U, "V": V, "W": W}

# Entry point required by evaluator
if __name__ == "__main__":
    # Just to ensure it's importable and callable
    pass
