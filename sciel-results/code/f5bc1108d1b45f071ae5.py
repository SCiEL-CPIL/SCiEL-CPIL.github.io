import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices
    # I: 0,1 (size 2), 2,3 (size 2)
    # J: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    # K: 0,1 (size 2), 2,3 (size 2), 4,5 (size 2)
    
    I_blocks = [
        [0, 1],
        [2, 3]
    ]
    J_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    K_blocks = [
        [0, 1],
        [2, 3],
        [4, 5]
    ]
    
    # Total rank calculation
    # 12 blocks of 2x2x2 (Rank 7)
    # 6 blocks of 2x1x2 (Rank 1)
    rank = 12 * 7 + 6 * 1
    rank = 90
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_offset = 0
    
    # Define Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for U and V
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) corresponding to C indices (k,i)
    
    strassen_u = [
        [1, 0, 0, 0], # P1: A00
        [1, 1, 0, 0], # P2: A00+A01
        [0, 0, 1, 0], # P3: A10
        [0, 0, 0, 1], # P4: A11
        [1, 0, 0, 1], # P5: A00+A11
        [0, 1, 0, 0], # P6: A01
        [0, 0, 1, 1]  # P7: A10+A11
    ]
    
    strassen_v = [
        [1, 0, 0, 0], # P1: B00
        [0, 0, 0, 1], # P2: B11
        [1, 0, 0, 0], # P3: B00
        [0, 1, 0, 0], # P4: B10
        [1, 0, 0, 1], # P5: B00+B11
        [0, 1, 0, 1], # P6: B10+B11
        [0, 1, 0, 0]  # P7: B01
    ]
    
    # Correct Strassen V vectors
    # P1 = A00 * (B00 - B11) -> v1 = [1, 0, 0, -1]
    # P2 = (A00+A01) * B11 -> v2 = [0, 0, 0, 1]
    # P3 = (A10-A00) * B00 -> v3 = [1, 0, 0, 0]
    # P4 = A11 * (B10-B00) -> v4 = [0, 0, 1, 0] ... wait B10 is (1,0), B00 is (0,0).
    # Indices for V: (j,k).
    # B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1).
    # P1: B00 - B11 -> [1, 0, 0, -1]
    # P2: B11 -> [0, 0, 0, 1]
    # P3: B00 -> [1, 0, 0, 0]
    # P4: B10 - B00 -> [0, 0, 1, 0] ... B10 is (1,0) which is index 2. B00 is (0,0) which is index 0.
    # So v4 = [-1, 0, 1, 0]
    # P5: B00 + B11 -> [1, 0, 0, 1]
    # P6: (A01-A11) * (B10+B11) -> v6 = [0, 0, 1, 1]
    # P7: (A10+A00) * (B00+B01) ... wait.
    # Standard Strassen:
    # P1 = A00(B00-B11)
    # P2 = (A00+A01)B11
    # P3 = (A10-A00)B00
    # P4 = A11(B10-B00)
    # P5 = (A00+A11)(B00+B11)
    # P6 = (A01-A11)(B10+B11)
    # P7 = (A10+A00)(B00+B01) ... No, usually (A10+A00) and (B00+B01) is not right.
    # P7 = (A10+A00) ... actually P7 = (A10+A00) is not standard.
    # P7 = (A00+A10) ... let's use the known correct set.
    
    # Correct Strassen V:
    strassen_v = [
        [1, 0, 0, -1], # P1
        [0, 0, 0, 1],  # P2
        [1, 0, 0, 0],  # P3
        [-1, 0, 1, 0], # P4 (B10 - B00)
        [1, 0, 0, 1],  # P5
        [0, 0, 1, 1],  # P6 (B10 + B11)
        [1, 1, 0, 0]   # P7 (B00 + B01)
    ]
    
    # Correct Strassen U:
    strassen_u = [
        [1, 0, 0, 0], # P1: A00
        [1, 1, 0, 0], # P2: A00+A01
        [-1, 0, 1, 0],# P3: -A00+A10
        [0, 0, 0, 1], # P4: A11
        [1, 0, 0, 1], # P5: A00+A11
        [0, 1, 0, -1],# P6: A01-A11
        [1, 0, 1, 0]  # P7: A00+A10
    ]
    
    # Correct Strassen W:
    # C00 = P5 + P4 - P2 - P7
    # C01 = P1 + P7
    # C10 = P3 + P5
    # C11 = P2 + P4 - P3 - P6
    
    # W indices (k,i):
    # C00 -> (0,0) index 0
    # C01 -> (1,0) index 2? No. K=0,1. I=0,1.
    # C_{i,k}.
    # C_{0,0} (i=0,k=0) -> W index (k,i) = (0,0) = 0
    # C_{0,1} (i=0,k=1) -> W index (k,i) = (1,0) = 2
    # C_{1,0} (i=1,k=0) -> W index (k,i) = (0,1) = 1
    # C_{1,1} (i=1,k=1) -> W index (k,i) = (1,1) = 3
    
    strassen_w = [
        [0, 0, 1, 0], # P1 -> C01 (idx 2)? No.
        # C01 = P1 + P7. So P1 contributes to C01. W_P1 has 1 at (1,0) which is index 2.
        # C00 = P5+P4-P2-P7.
        # C10 = P3+P5.
        # C11 = P2+P4-P3-P6.
        
        # P1: C01. W=[0,0,1,0]
        # P2: C00(-1), C11(+1). W=[-1,0,0,1]
        # P3: C10(+1), C11(-1). W=[0,1,0,-1]
        # P4: C00(+1), C11(+1). W=[1,0,0,1]
        # P5: C00(+1), C10(+1). W=[1,1,0,0]
        # P6: C11(-1). W=[0,0,0,-1]
        # P7: C00(-1), C01(+1). W=[-1,0,1,0]
        
        [0, 0, 1, 0], # P1
        [-1, 0, 0, 1],# P2
        [0, 1, 0, -1],# P3
        [1, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 0, -1],# P6
        [-1, 0, 1, 0] # P7
    ]

    def add_block(U_block, V_block, W_block, i_start, j_start, k_start, offset):
        for r in range(7):
            for a, i in enumerate(U_block[r]):
                if i != 0:
                    U[(i_start + a) * m + (j_start + 0), offset + r] += i
            for b, j in enumerate(V_block[r]):
                if j != 0:
                    V[(j_start + b) * p + (k_start + 0), offset + r] += j
            for c, k in enumerate(W_block[r]):
                if k != 0:
                    W[(k_start + c) * n + (i_start + 0), offset + r] += k

    # This simple add_block assumes 2x2 blocks.
    # We need to handle mapping of local indices to global indices.
    
    # Let's implement a more flexible block adder.
    
    def add_2x2x2_block(i0, j0, k0, offset):
        for r in range(7):
            # U part
            for li in range(2):
                for lj in range(2):
                    val = strassen_u[r][li*2+lj]
                    if val != 0:
                        U[(i0+li)*m + (j0+lj), offset+r] += val
            # V part
            for lj in range(2):
                for lk in range(2):
                    val = strassen_v[r][lj*2+lk]
                    if val != 0:
                        V[(j0+lj)*p + (k0+lk), offset+r] += val
            # W part
            for lk in range(2):
                for li in range(2):
                    val = strassen_w[r][lk*2+li]
                    if val != 0:
                        W[(k0+lk)*n + (i0+li), offset+r] += val
                        # Note: W index is (k,i).
                        # Global K is k0+lk. Global I is i0+li.
                        # W row index is k*n + i.
                        # So W[(k0+lk)*n + (i0+li), offset+r] += val

    def add_2x1x2_block(i0, j0, k0, offset):
        # Rank 1
        # u=1, v=1, w=1
        for li in range(2):
            U[(i0+li)*m + j0, offset] = 1
        V[j0*p + (k0+0), offset] = 1
        V[j0*p + (k0+1), offset] = 1
        for lk in range(2):
            for li in range(2):
                W[(k0+lk)*n + (i0+li), offset] = 1

    # Fill blocks
    # 12 blocks of 2x2x2
    for a in range(2):
        for b in range(2):
            for c in range(3):
                i0 = I_blocks[a][0]
                j0 = J_blocks[b][0]
                k0 = K_blocks[c][0]
                add_2x2x2_block(i0, j0, k0, col_offset)
                col_offset += 7

    # 6 blocks of 2x1x2
    for a in range(2):
        for c in range(3):
            i0 = I_blocks[a][0]
            j0 = J_blocks[2][0] # J_block 2 is size 1, index 4
            k0 = K_blocks[c][0]
            add_2x1x2_block(i0, j0, k0, col_offset)
            col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
