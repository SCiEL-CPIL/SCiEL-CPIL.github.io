import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank calculation:
    # 4 Strassen blocks (2x2x2) for top-left 2x2 A and first 4 cols of B (2x8 split into 4x2)
    # Wait, B is 2x8. Split into 4 blocks of 2x2.
    # Each 2x2x2 uses rank 7.
    # So 4 * 7 = 28.
    # Cross term A[0:2, 2] (2x1) * B[2, 0:8] (1x8) -> 2x8. Rank 16.
    # Cross term A[2, 0:2] (1x2) * B[0:2, 0:8] (2x8) -> 1x8. Rank 16.
    # Corner A[2, 2] (1x1) * B[2, 0:8] (1x8) -> 1x8. Rank 8.
    # Total rank = 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # A indices: i in {0,1}, j in {0,1} -> flat i*3+j
    # B indices: j in {0,1}, k in {k0, k0+1} -> flat j*8+k
    # C indices: k in {k0, k0+1}, i in {0,1} -> flat k*3+i
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen factors for 2x2x2
        # U (4x7), V (4x7), W (4x7)
        # Mapping:
        # U col r: coeffs of A
        # V col r: coeffs of B
        # W col r: coeffs of C
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U vectors (coeffs of A11, A12, A21, A22)
        # Indices in U: 
        # A11 is at (i_off, j_off)
        # A12 is at (i_off, j_off+1)
        # A21 is at (i_off+1, j_off)
        # A22 is at (i_off+1, j_off+1)
        
        u11 = i_off * m + j_off
        u12 = i_off * m + (j_off + 1)
        u21 = (i_off + 1) * m + j_off
        u22 = (i_off + 1) * m + (j_off + 1)
        
        # V vectors (coeffs of B11, B12, B21, B22)
        # B11 is at (j_off, k_off)
        # B12 is at (j_off, k_off+1)
        # B21 is at (j_off+1, k_off)
        # B22 is at (j_off+1, k_off+1)
        
        v11 = j_off * p + k_off
        v12 = j_off * p + (k_off + 1)
        v21 = (j_off + 1) * p + k_off
        v22 = (j_off + 1) * p + (k_off + 1)
        
        # W vectors (coeffs of C11, C12, C21, C22)
        # C11 is at (k_off, i_off)
        # C12 is at (k_off+1, i_off)
        # C21 is at (k_off, i_off+1)
        # C22 is at (k_off+1, i_off+1)
        
        w11 = k_off * n + i_off
        w12 = (k_off + 1) * n + i_off
        w21 = k_off * n + (i_off + 1)
        w22 = (k_off + 1) * n + (i_off + 1)
        
        # Strassen coefficients
        # r=0: M1
        U[u11, col] = 1; U[u22, col] = 1
        V[v11, col] = 1; V[v22, col] = 1
        W[w11, col] = 1; W[w22, col] = 1
        W[w12, col] = 0; W[w21, col] = 0
        # Correction: W is the output coefficients.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to accumulate W.
        # Let's define the W contributions for each M.
        
        # M1: C11 += 1, C22 += 1
        # M2: C21 += 1, C22 -= 1
        # M3: C12 += 1, C22 += 1
        # M4: C11 += 1, C21 += 1
        # M5: C11 -= 1, C12 += 1
        # M6: C22 += 1
        # M7: C11 += 1
        
        # r=0: M1
        U[u11, col] += 1; U[u22, col] += 1
        V[v11, col] += 1; V[v22, col] += 1
        W[w11, col] += 1; W[w22, col] += 1
        
        # r=1: M2
        col += 1
        U[u21, col] += 1; U[u22, col] += 1
        V[v11, col] += 1
        W[w21, col] += 1; W[w22, col] += -1
        
        # r=2: M3
        col += 1
        U[u11, col] += 1
        V[v12, col] += 1; V[v22, col] += -1
        W[w12, col] += 1; W[w22, col] += 1
        
        # r=3: M4
        col += 1
        U[u22, col] += 1
        V[v21, col] += 1; V[v11, col] += -1
        W[w11, col] += 1; W[w21, col] += 1
        
        # r=4: M5
        col += 1
        U[u11, col] += 1; U[u12, col] += 1
        V[v22, col] += 1
        W[w11, col] += -1; W[w12, col] += 1
        
        # r=5: M6
        col += 1
        U[u21, col] += 1; U[u11, col] += -1
        V[v11, col] += 1; V[v12, col] += 1
        W[w22, col] += 1
        
        # r=6: M7
        col += 1
        U[u12, col] += 1; U[u22, col] += -1
        V[v21, col] += 1; V[v22, col] += 1
        W[w11, col] += 1
        
        col += 1
        
    # Add 4 Strassen blocks for A[0:2, 0:2] and B[0:2, 0:8]
    # B cols 0-1, 2-3, 4-5, 6-7
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    
    # Add cross term A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
    # A[0,2], A[1,2]
    # B[2,0]...B[2,7]
    # C[0,0]...C[0,7], C[1,0]...C[1,7]
    # Rank 16
    for k in range(8):
        # A[0,2] * B[2,k] -> C[0,k]
        U[0 * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1
        # A[1,2] * B[2,k] -> C[1,k]
        U[1 * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1
        
    # Add cross term A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8]
    # A[2,0], A[2,1]
    # B[0,0]...B[0,7], B[1,0]...B[1,7]
    # C[2,0]...C[2,7]
    # Rank 16
    for k in range(8):
        # A[2,0] * B[0,k] -> C[2,k]
        U[2 * m + 0, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
        # A[2,1] * B[1,k] -> C[2,k]
        U[2 * m + 1, col] = 1
        V[1 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
        
    # Add corner A[2,2] * B[2,0:8] -> C[2,0:8]
    # A[2,2]
    # B[2,0]...B[2,7]
    # C[2,0]...C[2,7]
    # Rank 8
    for k in range(8):
        U[2 * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
        
    # Check col == rank
    # assert col == rank
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
