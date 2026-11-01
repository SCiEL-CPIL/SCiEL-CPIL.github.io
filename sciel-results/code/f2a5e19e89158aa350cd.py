"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Indices:
    # A: (i,j) -> i*m+j, i,j in {0,1} => 0,1,2,3
    # B: (j,k) -> j*p+k, j in {0,1}, k in {0..8} => 0..8 (j=0), 9..17 (j=1)
    # C: (k,i) -> k*n+i, k in {0..8}, i in {0,1} => 0..17
    
    # Known result: rank(2,2,p) = 4p-2 for p >= 3
    # Construction using 4p-2 = 34 terms:
    # 
    # We use the following decomposition:
    # For each k in 0..p-1:
    #   Term 1: A[0,0]*B[0,k] -> C[0,k]
    #   Term 2: A[1,0]*B[0,k] -> C[1,k]
    #   Term 3: (A[0,1]+A[1,1])*B[1,k] -> C[0,k] and C[1,k]
    #   Term 4: A[1,1]*(B[1,k]-B[0,k]) -> C[1,k]  [only for k < p-1, and adjusted for last k]
    #
    # Actually, let's use a cleaner known construction for rank 4p-2:
    # 
    # The key insight is that we can share terms across rows.
    # For (2,2,p), the minimal rank is 4p-2.
    
    rank = 4 * p - 2  # 34
    
    U = np.zeros((n * m, rank), dtype=float)  # (4, 34)
    V = np.zeros((m * p, rank), dtype=float)  # (18, 34)
    W = np.zeros((p * n, rank), dtype=float)  # (18, 34)
    
    col = 0
    
    # For k = 0 to p-2 (first 8 columns):
    # We use 4 terms per column, but the last column uses only 2 terms
    # Total: 4*(p-1) + 2 = 4p - 2
    
    for k in range(p - 1):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0, col] = 1      # A[0,0]
        V[0 * p + k, col] = 1  # B[0,k]
        W[k * n + 0, col] = 1  # C[0,k]
        col += 1
        
        # Term 2: A[1,0] * B[0,k] -> C[1,k]
        U[2, col] = 1      # A[1,0]
        V[0 * p + k, col] = 1  # B[0,k]
        W[k * n + 1, col] = 1  # C[1,k]
        col += 1
        
        # Term 3: (A[0,1] + A[1,1]) * B[1,k] -> C[0,k] and C[1,k]
        U[1, col] = 1      # A[0,1]
        U[3, col] = 1      # A[1,1]
        V[1 * p + k, col] = 1  # B[1,k]
        W[k * n + 0, col] = 1  # C[0,k]
        W[k * n + 1, col] = 1  # C[1,k]
        col += 1
        
        # Term 4: A[1,1] * (B[1,k] - B[0,k]) -> C[1,k]
        # This gives A[1,1]*B[1,k] to C[1,k] and -A[1,1]*B[0,k] to C[1,k]
        # Combined with term 2: A[1,0]*B[0,k] + A[1,1]*B[1,k] - A[1,1]*B[0,k]
        # = (A[1,0] - A[1,1])*B[0,k] + A[1,1]*B[1,k]
        # But we want A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # So this doesn't work directly. Let me reconsider.
        
        # Actually, the correct construction for rank 4p-2 for (2,2,p) is:
        # Let me look up the standard construction more carefully.
        pass
    
    # Let me use a verified construction instead.
    # Reset
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    # Verified construction for rank(2,2,p) = 4p - 2:
    # 
    # The idea: For each output column k, we have:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # We can write this as:
    # Let X_k = B[0,k], Y_k = B[1,k]
    # C[0,k] = A[0,0]*X_k + A[0,1]*Y_k
    # C[1,k] = A[1,0]*X_k + A[1,1]*Y_k
    #
    # We use the following 4p-2 terms:
    # For k = 0 to p-1:
    #   T1: A[0,0] * X_k -> C[0,k]
    #   T2: A[1,0] * X_k -> C[1,k]
    # For k = 0 to p-2:
    #   T3: (A[0,1] + A[1,1]) * Y_k -> C[0,k] + C[1,k]
    #   T4: (A[1,1] * X_k - A[0,1] * X_k) -> C[1,k] - C[0,k]  ... no this doesn't work
    
    # Let me try yet another approach. The known minimal rank for (2,2,p) is indeed 4p-2.
    # Here's a correct construction:
    
    # For k in 0..p-1:
    #   a_k = A[0,0], b_k = A[0,1], c_k = A[1,0], d_k = A[1,1]
    #   C[0,k] = a_k*B[0,k] + b_k*B[1,k]
    #   C[1,k] = c_k*B[0,k] + d_k*B[1,k]
    
    # Construction with 4p-2 terms:
    # For k = 0 to p-1:
    #   Term 2k:   A[0,0] * B[0,k] -> C[0,k]
    #   Term 2k+1: A[1,0] * B[0,k] -> C[1,k]
    # For k = 0 to p-2:
    #   Term 2p+k: (A[0,1]+A[1,1]) * B[1,k] -> C[0,k] and C[1,k]
    #   Term 2p+1+k: (A[1,1]-A[0,1]) * B[0,k] -> C[1,k] (negative for C[0,k]... no)
    
    # I'm going in circles. Let me just use the schoolbook rank 36 which is correct.
    rank = 4 * p  # 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
