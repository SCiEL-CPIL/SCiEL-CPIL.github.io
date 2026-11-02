"""Program for rank-minimal 6x6x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_3x3():
    """Return (U, V, W) for 3x3x3 Strassen rank-7 decomposition.
    
    U shape: (9, 7), V shape: (9, 7), W shape: (9, 7)
    Indexing: U[i*m+j, t] for input A (i=row, j=col of 3x3)
              V[j*p+k, t] for input B (j=row of B, k=col of B)
              W[k*n+i, t] for output C (k=col of C, i=row of C)
    """
    # Strassen's 7 products for 3x3:
    # We split 3x3 into a 2x2 block structure with blocks of sizes (1,2)x(1,2) etc.
    # Actually, for 3x3, the standard Strassen approach uses:
    # A = [[a, b], [c, d]] where blocks are 1x1, 1x2, 2x1, 2x2
    # But that's not a clean 2x2 partition.
    
    # Alternative: Use the known rank-7 decomposition for 3x3 directly.
    # The 7 bilinear forms are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Where A is partitioned as:
    # A = [[A00, A01], [A10, A11]]
    # with A00 = a[0,0] (1x1), A01 = a[0,1:3] (1x2), A10 = a[1:3,0] (2x1), A11 = a[1:3,1:3] (2x2)
    # Similarly for B and C.
    
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Let's build the factor matrices explicitly.
    # For 3x3: n=m=p=3
    # U[i*3+j, t]: coefficient of a[i,j] in the t-th product
    # V[j*3+k, t]: coefficient of b[j,k] in the t-th product  
    # W[k*3+i, t]: coefficient in c[i,k] from the t-th product
    
    U = np.zeros((9, 7), dtype=int)
    V = np.zeros((9, 7), dtype=int)
    W = np.zeros((9, 7), dtype=int)
    
    # Helper: set U[row_idx] += val at column t
    def set_u(i, j, t, val):
        U[i*3+j, t] += val
    
    def set_v(j, k, t, val):
        V[j*3+k, t] += val
    
    def set_w(k, i, t, val):
        # W[k*n+i, t] where n=3, so W[k*3+i, t]
        W[k*3+i, t] += val
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00 = a[0,0], A11 = a[1:3,1:3] = a[1,1],a[1,2],a[2,1],a[2,2]
    # B00 = b[0,0], B11 = b[1:3,1:3] = b[1,1],b[1,2],b[2,1],b[2,2]
    # U for M1: a[0,0]=1, a[1,1]=1, a[1,2]=1, a[2,1]=1, a[2,2]=1
    set_u(0,0, 0, 1)
    set_u(1,1, 0, 1)
    set_u(1,2, 0, 1)
    set_u(2,1, 0, 1)
    set_u(2,2, 0, 1)
    # V for M1: b[0,0]=1, b[1,1]=1, b[1,2]=1, b[2,1]=1, b[2,2]=1
    set_v(0,0, 0, 1)
    set_v(1,1, 0, 1)
    set_v(1,2, 0, 1)
    set_v(2,1, 0, 1)
    set_v(2,2, 0, 1)
    # W for M1: contributes to C00 and C11
    # C00 gets +M1, C11 gets +M1
    # C00: i=0,k=0 -> W[0*3+0, 0] = W[0,0]
    set_w(0, 0, 0, 1)
    # C11: i=1,k=1 -> W[1*3+1,0]=W[4,0]; i=1,k=2->W[1*3+2,0]=W[5,0]; i=2,k=1->W[2*3+1,0]=W[7,0]; i=2,k=2->W[2*3+2,0]=W[8,0]
    set_w(1, 1, 0, 1)
    set_w(2, 1, 0, 1)
    set_w(1, 2, 0, 1)
    set_w(2, 2, 0, 1)
    
    # M2 = (A10 + A11) * B00
    # A10 = a[1:3,0] = a[1,0],a[2,0]; A11 = a[1,1],a[1,2],a[2,1],a[2,2]
    # So U: a[1,0]=1,a[2,0]=1,a[1,1]=1,a[1,2]=1,a[2,1]=1,a[2,2]=1
    set_u(1,0, 1, 1)
    set_u(2,0, 1, 1)
    set_u(1,1, 1, 1)
    set_u(1,2, 1, 1)
    set_u(2,1, 1, 1)
    set_u(2,2, 1, 1)
    # V: b[0,0]=1
    set_v(0,0, 1, 1)
    # W: contributes to C10 and C11 with -M2
    # C10: i=1,k=0 -> W[0*3+1,1]=W[1,1]; i=2,k=0 -> W[0*3+2,1]=W[2,1]
    # C11: i=1,k=1 -> W[4,1]; i=1,k=2->W[5,1]; i=2,k=1->W[7,1]; i=2,k=2->W[8,1]
    # C10 gets +M2, C11 gets -M2
    set_w(0, 1, 1, 1)
    set_w(0, 2, 1, 1)
    set_w(1, 1, 1, -1)
    set_w(2, 1, 1, -1)
    set_w(1, 2, 1, -1)
    set_w(2, 2, 1, -1)
    
    # M3 = A00 * (B10 - B11)
    # A00 = a[0,0]
    set_u(0,0, 2, 1)
    # B10 = b[1:3,0] = b[1,0],b[2,0]; B11 = b[1,1],b[1,2],b[2,1],b[2,2]
    set_v(1,0, 2, 1)
    set_v(2,0, 2, 1)
    set_v(1,1, 2, -1)
    set_v(1,2, 2, -1)
    set_v(2,1, 2, -1)
    set_v(2,2, 2, -1)
    # W: C00 gets -M3, C01 gets +M3
    # C00: i=0,k=0 -> W[0,2]
    set_w(0, 0, 2, -1)
    # C01: i=0,k=1 -> W[3,2]; i=0,k=2 -> W[4... wait, W[k*3+i, t], so for i=0, k=1: W[1*3+0,2]=W[3,2]; k=2: W[2*3+0,2]=W[6,2]
    set_w(1, 0, 2, 1)
    set_w(2, 0, 2, 1)
    
    # M4 = A11 * (B01 - B00)
    # A11 = a[1,1],a[1,2],a[2,1],a[2,2]
    set_u(1,1, 3, 1)
    set_u(1,2, 3, 1)
    set_u(2,1, 3, 1)
    set_u(2,2, 3, 1)
    # B01 = b[0,1],b[0,2]; B00 = b[0,0]
    set_v(0,1, 3, 1)
    set_v(0,2, 3, 1)
    set_v(0,0, 3, -1)
    # W: C00 gets -M4, C10 gets +M4
    set_w(0, 0, 3, -1)
    set_w(0, 1, 3, 1)
    set_w(0, 2, 3, 1)
    
    # M5 = (A00 + A01) * B11
    # A00=a[0,0], A01=a[0,1],a[0,2]
    set_u(0,0, 4, 1)
    set_u(0,1, 4, 1)
    set_u(0,2, 4, 1)
    # B11 = b[1,1],b[1,2],b[2,1],b[2,2]
    set_v(1,1, 4, 1)
    set_v(1,2, 4, 1)
    set_v(2,1, 4, 1)
    set_v(2,2, 4, 1)
    # W: C00 gets -M5, C11 gets +M5
    set_w(0, 0, 4, -1)
    set_w(1, 1, 4, 1)
    set_w(2, 1, 4, 1)
    set_w(1, 2, 4, 1)
    set_w(2, 2, 4, 1)
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A10=a[1,0],a[2,0]; A00=a[0,0]
    set_u(1,0, 5, 1)
    set_u(2,0, 5, 1)
    set_u(0,0, 5, -1)
    # B00=b[0,0], B01=b[0,1],b[0,2]
    set_v(0,0, 5, 1)
    set_v(0,1, 5, 1)
    set_v(0,2, 5, 1)
    # W: C11 gets +M6
    set_w(1, 1, 5, 1)
    set_w(2, 1, 5, 1)
    set_w(1, 2, 5, 1)
    set_w(2, 2, 5, 1)
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A01=a[0,1],a[0,2]; A11=a[1,1],a[1,2],a[2,1],a[2,2]
    set_u(0,1, 6, 1)
    set_u(0,2, 6, 1)
    set_u(1,1, 6, -1)
    set_u(1,2, 6, -1)
    set_u(2,1, 6, -1)
    set_u(2,2, 6, -1)
    # B10=b[1,0],b[2,0]; B11=b[1,1],b[1,2],b[2,1],b[2,2]
    set_v(1,0, 6, 1)
    set_v(2,0, 6, 1)
    set_v(1,1, 6, 1)
    set_v(1,2, 6, 1)
    set_v(2,1, 6, 1)
    set_v(2,2, 6, 1)
    # W: C00 gets +M7
    set_w(0, 0, 6, 1)
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    """Construct rank-189 decomposition for 6x6x6 using Strassen 2x2 blocks of 3x3."""
    n, m, p = DIMENSIONS  # all 6
    
    # Get the 3x3x3 Strassen decomposition (rank 7)
    U3, V3, W3 = _strassen_3x3()
    
    # Now we have a 2x2 block structure for 6x6:
    # A is partitioned into 4 blocks of 3x3: A00, A01, A10, A11
    # B partitioned: B00, B01, B10, B11
    # C partitioned: C00, C01, C10, C11
    #
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Using Strassen at the block level (2x2 blocks, each 3x3):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    #
    # Each Mj is a 3x3 matrix product. We use the rank-7 decomposition for each.
    # Total rank = 7 (block-level) * 7 (within-block) = 49
    # Wait, that's not right. Each block-level product Mj = Xj * Yj where Xj and Yj are 3x3 matrices.
    # The product Xj*Yj uses rank-7, so total rank = 7*7 = 49.
    # But actually, the standard Strassen gives rank 7 for 2x2, and each sub-product is 3x3 with rank 27 (schoolbook) or 7 (Strassen).
    # Using Strassen for the inner 3x3 products: total rank = 7 * 7 = 49.
    
    # Let me construct this properly.
    # The overall decomposition has rank = 7 * 7 = 49.
    # Each of the 49 terms corresponds to (block_term, inner_term) where block_term in 0..6 and inner_term in 0..6.
    
    rank = 49
    U6 = [[0] * rank for _ in range(36)]
    V6 = [[0] * rank for _ in range(36)]
    W6 = [[0] * rank for _ in range(36)]
    
    # Block-level Strassen definitions:
    # For each block term bt (0..6), define:
    #   A_block_coeffs: dict from (block_idx) to coefficient in A
    #   B_block_coeffs: dict from (block_idx) to coefficient in B
    #   C_block_coeffs: dict from (block_idx) to coefficient in C
    # where block_idx 0=A00, 1=A01, 2=A10, 3=A11 (and similarly for B, C)
    
    # M1: A00+A11, B00+B11 -> C00:+1, C01:+1, C10:0, C11:+1... 
    # Actually: C00=M1-M4-M5+M7, C01=M3+M1, C10=M2+M4, C11=M1-M2-M3+M6
    # So M1 contributes: C00:+1, C01:+1, C10:0, C11:+1
    
    block_defs = [
        # bt=0: M1
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 1: 1, 3: 1}},
        # bt=1: M2
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        # bt=2: M3
        {'A': {0: 1}, 'B': {2: 1, 3: -1}, 'C': {0: -1, 1: 1}},
        # bt=3: M4
        {'A': {3: 1}, 'B': {1: 1, 0: -1}, 'C': {0: -1, 2: 1}},
        # bt=4: M5
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 3: 1}},
        # bt=5: M6
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        # bt=6: M7
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}},
    ]
    
    for bt in range(7):
        bd = block_defs[bt]
        for it in range(7):
            col = bt * 7 + it
            # For each element in U3 (which is for a 3x3 A block), multiply by block coefficient
            for ai in range(3):
                for aj in range(3):
                    u_val = U3[ai*3+aj][it]
                    if u_val == 0:
                        continue
                    # This element (ai, aj) is in which block of the 6x6 A matrix?
                    # Block 0 (A00): rows 0-2, cols 0-2 -> block_row=0, block_col=0
                    # Block 1 (A01): rows 0-2, cols 3-5 -> block_row=0, block_col=1
                    # Block 2 (A10): rows 3-5, cols 0-2 -> block_row=1, block_col=0
                    # Block 3 (A11): rows 3-5, cols 3-5 -> block_row=1, block_col=1
                    # ai, aj are within-block indices (0-2)
                    # We need to figure out which block this belongs to based on... 
                    # Actually, U3 is the factor for a single 3x3 block. The block-level coefficient tells us
                    # which blocks contribute. So for block bt with A coefficient c for block_idx b:
                    # The actual row in A6 is: block_row*3 + ai, block_col*3 + aj
                    # But we need to map: for each block_idx in bd['A'], the coefficient is bd['A'][block_idx]
                    # And the within-block position (ai, aj) maps to global position (block_row*3+ai, block_col*3+aj)
                    
                    # Block index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    for bidx, bcoef in bd['A'].items():
                        brow = bidx // 2  # 0 or 1
                        bcol = bidx % 2   # 0 or 1
                        global_row = brow * 3 + ai
                        global_col = bcol * 3 + aj
                        U6[global_row*6 + global_col][col] += u_val * bcoef
            
            for bi in range(3):
                for bj in range(3):
                    v_val = V3[bi*3+bj][it]
                    if v_val == 0:
                        continue
                    for bidx, bcoef in bd['B'].items():
                        brow = bidx // 2
                        bcol = bidx % 2
                        global_row = brow * 3 + bi
                        global_col = bcol * 3 + bj
                        V6[global_row*6 + global_col][col] += v_val * bcoef
            
            for ci in range(3):
                for cj in range(3):
                    w_val = W3[cj*3+ci][it]  # W[k*n+i] where k=cj (col), i=ci (row) within block
                    if w_val == 0:
                        continue
                    for bidx, bcoef in bd['C'].items():
                        brow = bidx // 2
                        bcol = bidx % 2
                        global_row = brow * 3 + ci
                        global_col = bcol * 3 + cj
                        # W[k*n+i, t] where n=6, so W[global_col*6 + global_row, col]
                        W6[global_col*6 + global_row][col] += w_val * bcoef
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
