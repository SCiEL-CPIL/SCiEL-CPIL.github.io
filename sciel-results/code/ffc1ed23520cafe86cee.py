"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_factors():
    """Return U, V, W for 2x2 x 2x2 multiplication (rank 7).
    
    U: (4, 7) for A indices (i, j)
    V: (4, 7) for B indices (j, k) 
    W: (4, 7) for C indices (k, i) -> W[k*2+i, r]
    """
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # A rows: 00, 01, 10, 11
    V = [[0]*7 for _ in range(4)]  # B cols: 00, 01, 10, 11
    W = [[0]*7 for _ in range(4)]  # C rows: W[k*2+i] for k,i in {0,1}
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 += M3, C11 += M3
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00 += M4, C10 += M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= M5, C01 += M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += M6
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 += M7
    W[0][6] = 1
    
    return U, V, W

def _build_4x4_factors():
    """Build rank-49 decomposition for 4x4 x 4x4 using recursive Strassen.
    
    Returns U: (16, 49), V: (16, 49), W: (16, 49)
    where W is indexed as W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    # For 4x4, we split into 2x2 blocks and apply Strassen recursively.
    # The 4x4 matrices have indices (i,j) where i,j in {0,1,2,3}
    # We group them as: block row i//2, block col j//2, local i%2, j%2
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r in range(49):
        r1 = r // 7  # which Strassen level-1 term
        r2 = r % 7   # which Strassen level-2 term
        
        for i1 in range(2):
            for j1 in range(2):
                if U2[i1*2 + j1][r1] != 0:
                    for i2 in range(2):
                        for j2 in range(2):
                            if U2[i2*2 + j2][r2] != 0:
                                # A index: global i = i1*2+i2, j = j1*2+j2
                                a_idx = (i1*2+i2)*4 + (j1*2+j2)
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                U4[a_idx][r] += val
        
        for j1 in range(2):
            for k1 in range(2):
                if V2[j1*2 + k1][r1] != 0:
                    for j2 in range(2):
                        for k2 in range(2):
                            if V2[j2*2 + k2][r2] != 0:
                                # B index: global row = j1*2+j2, col = k1*2+k2
                                b_idx = (j1*2+j2)*4 + (k1*2+k2)
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                V4[b_idx][r] += val
        
        for k1 in range(2):
            for i1 in range(2):
                if W2[k1*2 + i1][r1] != 0:
                    for k2 in range(2):
                        for i2 in range(2):
                            if W2[k2*2 + i2][r2] != 0:
                                # C index: W[k*4+i] where global k = k1*2+k2, i = i1*2+i2
                                c_idx = (k1*2+k2)*4 + (i1*2+i2)
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                W4[c_idx][r] += val
    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build 4x4 x 4x4 decomposition (rank 49)
    U4, V4, W4 = _build_4x4_factors()
    
    # For 4x4 x 4x8: split B into two 4x4 halves
    # B has rows j (0-3), cols k (0-7)
    # Left half: cols 0-3, Right half: cols 4-7
    # C has cols k (0-7), rows i (0-3)
    # C_left = A * B_left, C_right = A * B_right
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]  # (16, 98)
    V = [[0]*rank for _ in range(m*p)]  # (32, 98)
    W = [[0]*rank for _ in range(p*n)]  # (32, 98)
    
    # Copy U for both halves
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            U[i][r+49] = U4[i][r]
    
    # V: left half uses cols 0-3 of B, right half uses cols 4-7
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*8 + k][r] = V4[j*4 + k][r]
                V[j*8 + (k+4)][r+49] = V4[j*4 + k][r]
    
    # W: C has cols k (0-7), rows i (0-3)
    # W[k*n + i, r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4 + i][r] = W4[k*4 + i][r]
                W[(k+4)*4 + i][r+49] = W4[k*4 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

