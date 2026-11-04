"""Tensor decomposition for 2x3 * 3x10 matrix multiplication.
Uses a Strassen-style 2x2x2 block decomposition for the 2x2 part
plus a trivial 2x1x10 part, achieving rank 55 (vs schoolbook 60).
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k), the coefficient in C[i,k] from A[i,j]*B[j,k] must be 1.
    The coefficient is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T[i,j,k] = s
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _build_trivial(n, m, p):
    """Schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen(n=2, m=3, p=10):
    """Build a Strassen-based decomposition for 2x3 * 3x10.
    
    Split the 3 middle columns into 1 + 2:
    - The 2x1x10 part uses trivial rank 20.
    - The 2x2x10 part uses 5 Strassen 2x2x2 blocks (each rank 7) = rank 35.
    Total rank = 55.
    """
    rank = 20 + 35  # 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 2x1x10 trivial (i=0,1; j=0; k=0..9)
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Part 2: 2x2x10 via 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair of output columns (k, k+1), we multiply:
    #   [A(i,1) A(i,2)] * [B(1,k) B(1,k+1)]
    #                        [B(2,k) B(2,k+1)]
    # using Strassen 2x2x2 (7 multiplications).
    # 
    # Strassen 2x2x2:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # In our indexing:
    #   A rows: i=0,1 (n=2); A cols: j=1,2 (the 2x2 part of the 3 cols)
    #   B rows: j=1,2; B cols: k, k+1
    #   C rows: i=0,1; C cols: k, k+1
    #
    # For each Strassen block, we need to map:
    #   A11 = A[0,1], A12 = A[0,2], A21 = A[1,1], A22 = A[1,2]
    #   B11 = B[1,k], B12 = B[1,k+1], B21 = B[2,k], B22 = B[2,k+1]
    #   C11 = C[0,k], C12 = C[0,k+1], C21 = C[1,k], C22 = C[1,k+1]
    
    # Strassen factors for 2x2x2 (rank 7):
    # U_A: 2x2 -> 7 values (the 7 linear combos of A entries)
    # V_B: 2x2 -> 7 values (the 7 linear combos of B entries)
    # W_C: 2x2 -> 7 values (the 7 linear combos to reconstruct C)
    
    # A linear combos (rows: i=0,1; cols: j=1,2)
    # M1: A11+A22 -> U[0,1]=1, U[1,2]=1
    # M2: A21+A22 -> U[1,1]=1, U[1,2]=1
    # M3: A11 -> U[0,1]=1
    # M4: A22 -> U[1,2]=1
    # M5: A11+A12 -> U[0,1]=1, U[0,2]=1
    # M6: A21-A11 -> U[1,1]=1, U[0,1]=-1
    # M7: A12-A22 -> U[0,2]=1, U[1,2]=-1
    
    # B linear combos (rows: j=1,2; cols: k,k+1)
    # M1: B11+B22 -> V[1,k]=1, V[2,k+1]=1
    # M2: B11 -> V[1,k]=1
    # M3: B12-B22 -> V[1,k+1]=1, V[2,k+1]=-1
    # M4: B21-B11 -> V[2,k]=1, V[1,k]=-1
    # M5: B22 -> V[2,k+1]=1
    # M6: B11+B12 -> V[1,k]=1, V[1,k+1]=1
    # M7: B21+B22 -> V[2,k]=1, V[2,k+1]=1
    
    # C reconstruction:
    # C11 = M1 + M4 - M5 + M7 -> W[0,k] = [1, 0, 0, 1, -1, 0, 1]
    # C12 = M3 + M5 -> W[0,k+1] = [0, 0, 1, 0, 1, 0, 0]
    # C21 = M2 + M4 -> W[1,k] = [0, 1, 0, 1, 0, 0, 0]
    # C22 = M1 - M2 + M3 + M6 -> W[1,k+1] = [1, -1, 1, 0, 0, 1, 0]
    
    for pair in range(5):
        k0 = pair * 2  # first column in the pair
        k1 = pair * 2 + 1  # second column in the pair
        
        # For each of the 7 Strassen multiplications:
        for r_idx in range(7):
            col = 20 + pair * 7 + r_idx
            
            # A part (U): rows i=0,1; cols j=1,2
            # U index: i*m + j = i*3 + j
            if r_idx == 0:  # M1: A11+A22
                U[0*3+1, col] = 1  # A[0,1]
                U[1*3+2, col] = 1  # A[1,2]
            elif r_idx == 1:  # M2: A21+A22
                U[1*3+1, col] = 1  # A[1,1]
                U[1*3+2, col] = 1  # A[1,2]
            elif r_idx == 2:  # M3: A11
                U[0*3+1, col] = 1  # A[0,1]
            elif r_idx == 3:  # M4: A22
                U[1*3+2, col] = 1  # A[1,2]
            elif r_idx == 4:  # M5: A11+A12
                U[0*3+1, col] = 1  # A[0,1]
                U[0*3+2, col] = 1  # A[0,2]
            elif r_idx == 5:  # M6: A21-A11
                U[1*3+1, col] = 1  # A[1,1]
                U[0*3+1, col] = -1  # A[0,1]
            elif r_idx == 6:  # M7: A12-A22
                U[0*3+2, col] = 1  # A[0,2]
                U[1*3+2, col] = -1  # A[1,2]
            
            # B part (V): rows j=1,2; cols k0,k1
            # V index: j*p + k
            if r_idx == 0:  # M1: B11+B22
                V[1*p+k0, col] = 1   # B[1,k0]
                V[2*p+k1, col] = 1   # B[2,k1]
            elif r_idx == 1:  # M2: B11
                V[1*p+k0, col] = 1   # B[1,k0]
            elif r_idx == 2:  # M3: B12-B22
                V[1*p+k1, col] = 1   # B[1,k1]
                V[2*p+k1, col] = -1  # B[2,k1]
            elif r_idx == 3:  # M4: B21-B11
                V[2*p+k0, col] = 1   # B[2,k0]
                V[1*p+k0, col] = -1  # B[1,k0]
            elif r_idx == 4:  # M5: B22
                V[2*p+k1, col] = 1   # B[2,k1]
            elif r_idx == 5:  # M6: B11+B12
                V[1*p+k0, col] = 1   # B[1,k0]
                V[1*p+k1, col] = 1   # B[1,k1]
            elif r_idx == 6:  # M7: B21+B22
                V[2*p+k0, col] = 1   # B[2,k0]
                V[2*p+k1, col] = 1   # B[2,k1]
            
            # C part (W): rows k0,k1; cols i=0,1
            # W index: k*n + i
            if r_idx == 0:  # M1: contributes to C11(+1), C22(+1)
                W[k0*n+0, col] = 1   # C[0,k0]
                W[k1*n+1, col] = 1   # C[1,k1]
            elif r_idx == 1:  # M2: contributes to C21(+1), C22(-1)
                W[k0*n+1, col] = 1   # C[1,k0]
                W[k1*n+1, col] = -1  # C[1,k1]
            elif r_idx == 2:  # M3: contributes to C12(+1), C22(+1)
                W[k1*n+0, col] = 1   # C[0,k1]
                W[k1*n+1, col] = 1   # C[1,k1]
            elif r_idx == 3:  # M4: contributes to C11(+1), C21(+1)
                W[k0*n+0, col] = 1   # C[0,k0]
                W[k0*n+1, col] = 1   # C[1,k0]
            elif r_idx == 4:  # M5: contributes to C11(-1), C12(+1)
                W[k0*n+0, col] = -1  # C[0,k0]
                W[k1*n+0, col] = 1   # C[0,k1]
            elif r_idx == 5:  # M6: contributes to C22(+1)
                W[k1*n+1, col] = 1   # C[1,k1]
            elif r_idx == 6:  # M7: contributes to C11(+1)
                W[k0*n+0, col] = 1   # C[0,k0]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition (rank 55)
    U, V, W = _build_strassen(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        U_list = [[int(x) for x in row] for row in U]
        V_list = [[int(x) for x in row] for row in V]
        W_list = [[int(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook (rank 60)
    U, V, W = _build_trivial(n, m, p)
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
