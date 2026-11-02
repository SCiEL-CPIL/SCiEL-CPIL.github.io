import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 100. We aim for 92.
    # Block decomposition:
    # A: 4x5 -> [2x2, 2x2, 2x1] (rows 0-1, 2-3; cols 0-1, 2-3, 4)
    # B: 5x5 -> [2x2, 2x2, 2x1] (rows 0-1, 2-3, 4; cols 0-1, 2-3, 4)
    
    # Total Rank 92
    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A indices: i in {0,1} (global row offset), j in {0,1} (global col offset)
    # B indices: j in {0,1} (global row offset), k in {0,1} (global col offset)
    # C indices: i in {0,1}, k in {0,1}
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # Strassen multiplications
        # M1 = (A00+A11)(B00+B11)
        U[i_off*2 + 0, col] += 1
        U[i_off*2 + 1, col] += 1
        V[j_off*2 + 0, col] += 1
        V[j_off*2 + 1, col] += 1
        W[k_off*2 + 0, i_off*2 + 0] += 1
        W[k_off*2 + 0, i_off*2 + 1] += 1
        W[k_off*2 + 1, i_off*2 + 0] += 1
        W[k_off*2 + 1, i_off*2 + 1] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[i_off*2 + 1, col] += 1 # A10
        U[i_off*2 + 1, col] += 1 # A11 -- Wait, U[i,j] is A.
        # A10 is at row i_off*2+1, col j_off*2+0.
        # A11 is at row i_off*2+1, col j_off*2+1.
        # So U[i_off*2+1, j_off*2+0] and U[i_off*2+1, j_off*2+1]?
        # No, U is n*m x rank. U[row, col].
        # A is n x m.
        # U[i*m+j, c].
        
        # Let's restart the helper with correct indexing.
        pass

    # Correct Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # Re-initialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[i_off*2 + 0, col] += 1 # A00
        U[i_off*2 + 1, col] += 1 # A11
        V[j_off*2 + 0, col] += 1 # B00
        V[j_off*2 + 1, col] += 1 # B11
        W[k_off*2 + 0, i_off*2 + 0] += 1 # C00
        W[k_off*2 + 0, i_off*2 + 1] += 1 # C01? No.
        # W[k*n + i, c].
        # C00 corresponds to k=0, i=0. W[0*n + 0, c].
        # C01 corresponds to k=1, i=0. W[1*n + 0, c].
        # C10 corresponds to k=0, i=1. W[0*n + 1, c].
        # C11 corresponds to k=1, i=1. W[1*n + 1, c].
        
        # M1 contributes to C00, C01, C10, C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 adds to C00 and C11.
        W[k_off*n + i_off*2 + 0, col] += 1 # C00 (k=0, i=0)
        W[k_off*n + i_off*2 + 1, col] += 1 # C11? No.
        # C11 is k=1, i=1.
        # W index: k*n + i.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> n
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> n+1
        
        # M1: C00 and C11.
        W[0*n + 0, col] += 1 # Local C00
        W[1*n + 1, col] += 1 # Local C11
        # But we are in a block, so we need offsets.
        # Local k=0 -> Global k = k_off*2
        # Local k=1 -> Global k = k_off*2 + 1
        # Local i=0 -> Global i = i_off*2
        # Local i=1 -> Global i = i_off*2 + 1
        
        # Global W index for Local C00: (k_off*2)*n + (i_off*2)
        # Global W index for Local C11: (k_off*2 + 1)*n + (i_off*2 + 1)
        
        W[(k_off*2)*n + (i_off*2), col] += 1
        W[(k_off*2 + 1)*n + (i_off*2 + 1), col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[(i_off*2+1)*m + (j_off*2+0), col] += 1 # A10
        U[(i_off*2+1)*m + (j_off*2+1), col] += 1 # A11
        V[(j_off*2+0)*p + (k_off*2+0), col] += 1 # B00
        # C10 = M2 + M4
        W[(k_off*2)*n + (i_off*2+1), col] += 1
        col += 1

        # M3 = A00(B01-B11)
        U[(i_off*2+0)*m + (j_off*2+0), col] += 1 # A00
        V[(j_off*2+0)*p + (k_off*2+1), col] += 1 # B01
        V[(j_off*2+1)*p + (k_off*2+1), col] -= 1 # B11
        # C01 = M3 + M5
        W[(k_off*2+1)*n + (i_off*2+0), col] += 1
        col += 1

        # M4 = A11(B10-B00)
        U[(i_off*2+1)*m + (j_off*2+1), col] += 1 # A11
        V[(j_off*2+1)*p + (k_off*2+0), col] += 1 # B10
        V[(j_off*2+0)*p + (k_off*2+0), col] -= 1 # B00
        # C00 += M4, C10 += M4
        W[(k_off*2)*n + (i_off*2+0), col] += 1
        W[(k_off*2)*n + (i_off*2+1), col] += 1
        col += 1

        # M5 = (A00+A01)B11
        U[(i_off*2+0)*m + (j_off*2+0), col] += 1 # A00
        U[(i_off*2+0)*m + (j_off*2+1), col] += 1 # A01
        V[(j_off*2+1)*p + (k_off*2+1), col] += 1 # B11
        # C01 += M5, C00 -= M5
        W[(k_off*2+1)*n + (i_off*2+0), col] += 1
        W[(k_off*2)*n + (i_off*2+0), col] -= 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[(i_off*2+1)*m + (j_off*2+0), col] += 1 # A10
        U[(i_off*2+0)*m + (j_off*2+0), col] -= 1 # A00
        V[(j_off*2+0)*p + (k_off*2+0), col] += 1 # B00
        V[(j_off*2+0)*p + (k_off*2+1), col] += 1 # B01
        # C11 += M6
        W[(k_off*2+1)*n + (i_off*2+1), col] += 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[(i_off*2+0)*m + (j_off*2+1), col] += 1 # A01
        U[(i_off*2+1)*m + (j_off*2+1), col] -= 1 # A11
        V[(j_off*2+1)*p + (k_off*2+0), col] += 1 # B10
        V[(j_off*2+1)*p + (k_off*2+1), col] += 1 # B11
        # C00 += M7
        W[(k_off*2)*n + (i_off*2+0), col] += 1
        col += 1

    def add_schoolbook_block(i_off, j_off, k_off, h, w, d):
        # h: rows of A block, w: cols of A block (rows of B block), d: cols of B block
        # A is h x w, B is w x d, C is h x d
        nonlocal col
        for ii in range(h):
            for ww in range(w):
                for dd in range(d):
                    U[(i_off+ii)*m + (j_off+ww), col] = 1
                    V[(j_off+ww)*p + (k_off+dd), col] = 1
                    W[(k_off+dd)*n + (i_off+ii), col] = 1
                    col += 1

    # 8 blocks of 2x2x2
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                add_strassen_2x2x2(i_off, j_off, k_off)
                
    # j=2 terms (A is 2x1, B is 1x2 or 1x1)
    # i in {0, 2} (offsets for 2 rows)
    # j=4 (offset for 1 col)
    for i_off in [0, 2]:
        # k=0,1 (B is 1x2)
        add_schoolbook_block(i_off, 4, 0, 2, 1, 2)
        add_schoolbook_block(i_off, 4, 2, 2, 1, 2)
        # k=2 (B is 1x1)
        add_schoolbook_block(i_off, 4, 4, 2, 1, 1)
        
    # k=4 terms (A is 2x2, B is 2x1)
    # i in {0, 2}, j in {0, 2}
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            add_schoolbook_block(i_off, j_off, 4, 2, 2, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
