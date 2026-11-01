import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Known best rank is 28
    # We construct the rank 28 solution directly.
    # 
    # The tensor T_{i,j,k} decomposes into a sum of rank-1 tensors.
    # We use a block-based approach:
    # Block A: Top-left 2x2x2 (i in 0,1; j in 0,1; k in 0,1) -> Rank 7 (Strassen)
    # Block B: Top-right 2x2x2 (i in 0,1; j in 0,1; k in 2,3) -> Rank 7 (Strassen)
    # Block C: Top-middle 2x2x1 (i in 0,1; j in 0,1; k in 4)  -> Rank 4 (Schoolbook)
    # Block D: Bottom 2x1x5   (i in 0,1; j in 2;    k in 0..4) -> Rank 10 (Schoolbook)
    # Total Rank = 7 + 7 + 4 + 10 = 28
    
    R = 28
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    
    # Indices
    # U is indexed by (i*m + j). i in [0,1], j in [0,2]
    # V is indexed by (j*p + k). j in [0,2], k in [0,4]
    # W is indexed by (k*n + i). k in [0,4], i in [0,1]
    
    def set_U(i, j, r, val):
        U[i*m + j, r] += val
        
    def set_V(j, k, r, val):
        V[j*p + k, r] += val
        
    def set_W(k, i, r, val):
        W[k*n + i, r] += val

    # --- Block A: Strassen 2x2x2 for k=0,1 ---
    # Standard Strassen terms for matrices A (2x2) and B (2x2)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = A11(B10-B00) -> C10 += M2
    # M3 = (A00-A11)(B01+B11) -> C01 += M3
    # M4 = (A10+A11)B11 -> C11 -= M4
    # M5 = A00(B01-B11) -> C01 -= M5
    # M6 = (A01-A10)(B00+B10) -> C10 -= M6
    # M7 = (A01+A10)(B11-B10) -> C00 -= M7
    
    # Map to our indices: i is row of A, j is col of A / row of B, k is col of B
    # A[i,j] corresponds to U index. B[j,k] corresponds to V index. C[i,k] corresponds to W index.
    
    # Term 1: M1 = (U_{0,0}+U_{1,1}) * (V_{0,0}+V_{1,1}) -> W_{0,0} += M1, W_{1,1} += M1
    # Note: The "Strassen" decomposition here is applied to the 2x2 sub-tensor.
    # We define the rank-1 terms explicitly.
    
    # Rank 7 decomposition of 2x2x2 tensor T_{i,j,k} = A_{ij} B_{jk}
    # The 7 vectors are:
    # 1. u1=(1,0,0,1), v1=(1,0,0,1), w1=(1,0,0,1)  -- Note: u is (A00,A01,A10,A11), v is (B00,B01,B10,B11)
    #    Wait, standard Strassen:
    #    M1 = (A11+A00)(B11+B00)
    #    M2 = A11(B10-B00)
    #    M3 = (A00-A11)(B01+B11)
    #    M4 = (A10+A11)B11
    #    M5 = A00(B01-B11)
    #    M6 = (A01-A10)(B00+B10)
    #    M7 = (A01+A10)(B11-B10)
    
    # Let's expand these into the U, V, W components for the sub-block.
    # Sub-block indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Global indices: i=i', j=j', k=k' for Block A.
    
    # M1:
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M2:
    set_U(1, 1, 1, 1)
    set_V(1, 0, 1, 1); set_V(0, 0, 1, -1)
    set_W(0, 1, 1, 1)
    
    # M3:
    set_U(0, 0, 2, 1); set_U(1, 1, 2, -1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, 1)
    set_W(1, 0, 2, 1)
    
    # M4:
    set_U(1, 0, 3, 1); set_U(1, 1, 3, 1)
    set_V(1, 1, 3, 1)
    set_W(1, 1, 3, -1)
    
    # M5:
    set_U(0, 0, 4, 1)
    set_V(0, 1, 4, 1); set_V(1, 1, 4, -1)
    set_W(1, 0, 4, -1)
    
    # M6:
    set_U(0, 1, 5, 1); set_U(1, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(1, 0, 5, 1)
    set_W(0, 1, 5, -1)
    
    # M7:
    set_U(0, 1, 6, 1); set_U(1, 0, 6, 1)
    set_V(1, 1, 6, 1); set_V(1, 0, 6, -1)
    set_W(0, 0, 6, -1)

    # --- Block B: Strassen 2x2x2 for k=2,3 ---
    # Global indices: i in {0,1}, j in {0,1}, k in {2,3}
    # Local k' = k - 2.
    
    # M1:
    set_U(0, 0, 7, 1); set_U(1, 1, 7, 1)
    set_V(0, 2, 7, 1); set_V(1, 3, 7, 1)  # B00_local = B02_global, B11_local = B13_global
    set_W(2, 0, 7, 1); set_W(3, 1, 7, 1)
    
    # M2:
    set_U(1, 1, 8, 1)
    set_V(1, 2, 8, 1); set_V(0, 2, 8, -1) # B10_local = B12_global, B00_local = B02_global
    set_W(2, 1, 8, 1)
    
    # M3:
    set_U(0, 0, 9, 1); set_U(1, 1, 9, -1)
    set_V(0, 3, 9, 1); set_V(1, 3, 9, 1)  # B01_local = B03_global, B11_local = B13_global
    set_W(3, 0, 9, 1)
    
    # M4:
    set_U(1, 0, 10, 1); set_U(1, 1, 10, 1)
    set_V(1, 3, 10, 1)
    set_W(3, 1, 10, -1)
    
    # M5:
    set_U(0, 0, 11, 1)
    set_V(0, 3, 11, 1); set_V(1, 3, 11, -1)
    set_W(3, 0, 11, -1)
    
    # M6:
    set_U(0, 1, 12, 1); set_U(1, 0, 12, -1)
    set_V(0, 2, 12, 1); set_V(1, 2, 12, 1)
    set_W(2, 1, 12, -1)
    
    # M7:
    set_U(0, 1, 13, 1); set_U(1, 0, 13, 1)
    set_V(1, 3, 13, 1); set_V(1, 2, 13, -1)
    set_W(2, 0, 13, -1)

    # --- Block C: Schoolbook 2x2x1 for k=4 ---
    # Global indices: i in {0,1}, j in {0,1}, k=4
    # 4 terms:
    # C[i,4] += A[i,0] B[0,4]
    # C[i,4] += A[i,1] B[1,4]
    
    # Term 1: A[0,0] B[0,4] -> W[4,0]
    set_U(0, 0, 14, 1)
    set_V(0, 4, 14, 1)
    set_W(4, 0, 14, 1)
    
    # Term 2: A[1,0] B[0,4] -> W[4,1]
    set_U(1, 0, 15, 1)
    set_V(0, 4, 15, 1)
    set_W(4, 1, 15, 1)
    
    # Term 3: A[0,1] B[1,4] -> W[4,0]
    set_U(0, 1, 16, 1)
    set_V(1, 4, 16, 1)
    set_W(4, 0, 16, 1)
    
    # Term 4: A[1,1] B[1,4] -> W[4,1]
    set_U(1, 1, 17, 1)
    set_V(1, 4, 17, 1)
    set_W(4, 1, 17, 1)

    # --- Block D: Schoolbook 2x1x5 for j=2 ---
    # Global indices: i in {0,1}, j=2, k in {0,1,2,3,4}
    # 10 terms:
    # C[i,k] += A[i,2] B[2,k]
    
    # i=0
    for k in range(5):
        r = 18 + k
        set_U(0, 2, r, 1)
        set_V(2, k, r, 1)
        set_W(k, 0, r, 1)
        
    # i=1
    for k in range(5):
        r = 23 + k
        set_U(1, 2, r, 1)
        set_V(2, k, r, 1)
        set_W(k, 1, r, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
