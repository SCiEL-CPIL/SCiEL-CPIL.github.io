import numpy as np

DIMENSIONS = [2, 4, 4]

# Pre-computed entries for a rank 28 decomposition of 2x4 * 4x4 matrix multiplication.
# Based on applying Strassen's algorithm to the four 2x2 blocks.
# U: (8, 28), V: (16, 28), W: (8, 28)
# Format: List of (index, coefficient) pairs for each column (rank step)

def _generate_strassen_decomposition():
    # Strassen terms for 2x2 * 2x2 -> 2x2
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_cols = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        {0: [ (0,1), (3,1) ], 1: [ (0,1), (3,1) ], 2: [ (0,1), (3,1) ]},
        # M2: (A10+A11)B00 -> C10-C11
        {0: [ (2,1), (3,1) ], 1: [ (0,1) ], 2: [ (2,1), (3,-1) ]},
        # M3: A00(B01-B11) -> C01+C11
        {0: [ (0,1) ], 1: [ (1,1), (3,-1) ], 2: [ (1,1), (3,1) ]},
        # M4: A11(B10-B00) -> C00+C10
        {0: [ (3,1) ], 1: [ (2,1), (0,-1) ], 2: [ (0,1), (2,1) ]},
        # M5: (A00+A01)B11 -> -C00+C01
        {0: [ (0,1), (1,1) ], 1: [ (3,1) ], 2: [ (0,-1), (1,1) ]},
        # M6: (A10-A00)(B00+B01) -> C11
        {0: [ (2,1), (0,-1) ], 1: [ (0,1), (1,1) ], 2: [ (3,1) ]},
        # M7: (A01-A11)(B10+B11) -> C00
        {0: [ (1,1), (3,-1) ], 1: [ (2,1), (3,1) ], 2: [ (0,1) ]}
    ]
    
    # We need to map this to 2x4 * 4x4.
    # A is 2x4 (indices 0-7). Split into A_L (0-3) and A_R (4-7).
    # B is 4x4 (indices 0-15). Split into B_TL(0-3), B_TR(4-7), B_BL(8-11), B_BR(12-15).
    # C is 2x4 (indices 0-7 in output convention W[k*n+i]).
    # Output C is split into C_L (cols 0,1) and C_R (cols 2,3).
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    def map_A(local_idx, offset):
        # local_idx 0-3 corresponds to A's 2x2 block.
        # offset 0 -> A_L (cols 0,1 of A), offset 1 -> A_R (cols 2,3 of A).
        # A index = row * 4 + col.
        # Local (r,c) -> Global (r, offset*2 + c).
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (offset * 2 + c)
        
    def map_B(local_idx, row_off, col_off):
        # Local (r,c) -> Global (row_off*2 + r, col_off*2 + c).
        r = local_idx // 2
        c = local_idx % 2
        return (row_off * 2 + r) * 4 + (col_off * 2 + c)
        
    def map_C(local_idx, col_off):
        # Output C is 2x4.
        # Local C is 2x2 for a specific pair of output columns.
        # Output convention: W[k*n + i] where k is col, i is row.
        # Local C col (0,1) corresponds to global col (col_off*2, col_off*2+1).
        # Local (r,c) -> Global (r, col_off*2 + c).
        # Global Index = (col_off*2 + c) * 2 + r.
        r = local_idx // 2
        c = local_idx % 2
        return (col_off * 2 + c) * 2 + r

    # Block 1: A_L * B_TL -> C_L (part 1)
    for j, col_def in enumerate(strassen_cols):
        col_idx = j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 0, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 0), col_idx] += coeff
            
    # Block 2: A_R * B_BL -> C_L (part 2)
    for j, col_def in enumerate(strassen_cols):
        col_idx = 7 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 1, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 0), col_idx] += coeff
            
    # Block 3: A_L * B_TR -> C_R (part 1)
    for j, col_def in enumerate(strassen_cols):
        col_idx = 14 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 0, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 1), col_idx] += coeff
            
    # Block 4: A_R * B_BR -> C_R (part 2)
    for j, col_def in enumerate(strassen_cols):
        col_idx = 21 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 1, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 1), col_idx] += coeff
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _generate_strassen_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
