import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We'll use a block decomposition approach:
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x8) = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    
    # For each 2x2 * 2x8 multiplication, we split B into 4 blocks of 2x2
    # and use Strassen (rank 7 per 2x2 block), giving 7*4 = 28 terms per block
    # Total rank = 28 + 28 = 56
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=float)  # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=float)  # 16 x 56
    
    # Strassen 2x2 * 2x2 decomposition (rank 7)
    # Local indices:
    # U_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Standard Strassen formulas:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b11-b01)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    # Global mapping for A1 (cols 0,1 of A): local (i,j) -> global i*4+j
    a1_map = [0, 1, 4, 5]
    # Global mapping for A2 (cols 2,3 of A): local (i,j) -> global i*4+2+j
    a2_map = [2, 3, 6, 7]
    
    # Process A1*B1 (B rows 0,1)
    for c_pair in range(4):
        base_r = c_pair * 7
        # B1 is rows 0,1 of global B. Each row has 8 columns.
        # For column pair c_pair, we use columns 2*c_pair and 2*c_pair+1
        b1_map = [
            0*8 + 2*c_pair,       # row 0, col 2*c_pair
            0*8 + 2*c_pair + 1,   # row 0, col 2*c_pair+1
            1*8 + 2*c_pair,       # row 1, col 2*c_pair
            1*8 + 2*c_pair + 1    # row 1, col 2*c_pair+1
        ]
        
        # C1 output: rows 0,1 of C, columns 2*c_pair, 2*c_pair+1
        # W convention: W[k*n+i, r] where n=2
        # So W index = k*2 + i
        c1_map = [
            (2*c_pair)*2 + 0,       # k=2*c_pair, i=0
            (2*c_pair+1)*2 + 0,     # k=2*c_pair+1, i=0
            (2*c_pair)*2 + 1,       # k=2*c_pair, i=1
            (2*c_pair+1)*2 + 1      # k=2*c_pair+1, i=1
        ]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a1_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b1_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c1_map[lc], r] += val
    
    # Process A2*B2 (B rows 2,3)
    for c_pair in range(4):
        base_r = 28 + c_pair * 7
        # B2 is rows 2,3 of global B.
        b2_map = [
            2*8 + 2*c_pair,       # row 2, col 2*c_pair
            2*8 + 2*c_pair + 1,   # row 2, col 2*c_pair+1
            3*8 + 2*c_pair,       # row 3, col 2*c_pair
            3*8 + 2*c_pair + 1    # row 3, col 2*c_pair+1
        ]
        
        # C2 output: same as C1 (they add up)
        c2_map = [
            (2*c_pair)*2 + 0,       # k=2*c_pair, i=0
            (2*c_pair+1)*2 + 0,     # k=2*c_pair+1, i=0
            (2*c_pair)*2 + 1,       # k=2*c_pair, i=1
            (2*c_pair+1)*2 + 1      # k=2*c_pair+1, i=1
        ]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a2_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b2_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c2_map[lc], r] += val
    
    # Verify the full decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(total_rank):
                    actual += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(actual - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 64
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m + jj, idx] = 1.0
                    V[jj*p + kk, idx] = 1.0
                    W[kk*n + ii, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - start_time > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
