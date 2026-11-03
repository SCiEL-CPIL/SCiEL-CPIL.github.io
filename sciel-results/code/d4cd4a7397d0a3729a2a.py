import time
import sys

DIMENSIONS = [2, 4, 4]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We decompose C = A * B
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # Split B into B_11 (2x2, rows 0-1, cols 0-1), B_12 (2x2, rows 0-1, cols 2-3),
    #            B_21 (2x2, rows 2-3, cols 0-1), B_22 (2x2, rows 2-3, cols 2-3)
    # C_L = A_L * B_11 + A_R * B_21 (2x2)
    # C_R = A_L * B_12 + A_R * B_22 (2x2)
    # C = [C_L, C_R]
    
    # Each 2x2 * 2x2 product is done via Strassen (rank 7).
    # We have 4 such products.
    # Product 1: A_L * B_11 contributes to C_L
    # Product 2: A_R * B_21 contributes to C_L
    # Product 3: A_L * B_12 contributes to C_R
    # Product 4: A_R * B_22 contributes to C_R
    
    # Indexing conventions:
    # A[i, j] -> U[i*m + j, r]
    # B[i, j] -> V[i*p + j, r]
    # C[i, j] -> W[j*n + i, r]  (Output factor follows W[k*n+i,r] where k is col, i is row)
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        """
        Adds Strassen decomposition for A_block * B_block -> C_block
        A_block is 2x2 at A[A_r0:A_r0+2, A_c0:A_c0+2]
        B_block is 2x2 at B[B_r0:B_r0+2, B_c0:B_c0+2]
        C_block is 2x2 at C[C_r0:C_r0+2, C_c0:C_c0+2]
        """
        # A indices
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        # B indices
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        # C indices (C[row, col] -> W[col*n + row])
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r

    # Product 1: A_L (rows 0-1, cols 0-1) * B_11 (rows 0-1, cols 0-1) -> C_L (rows 0-1, cols 0-1)
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Product 2: A_R (rows 0-1, cols 2-3) * B_21 (rows 2-3, cols 0-1) -> C_L (rows 0-1, cols 0-1)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    
    # Product 3: A_L (rows 0-1, cols 0-1) * B_12 (rows 0-1, cols 2-3) -> C_R (rows 0-1, cols 2-3)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    
    # Product 4: A_R (rows 0-1, cols 2-3) * B_22 (rows 2-3, cols 2-3) -> C_R (rows 0-1, cols 2-3)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    
    return {"U": U, "V": V, "W": W}
