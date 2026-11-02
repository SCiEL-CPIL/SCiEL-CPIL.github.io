"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split the 2x3x8 multiplication into blocks.
    # We split the middle dimension m=3 into two parts: m1=2 and m2=1.
    # We split the right dimension p=8 into two parts: p1=4 and p2=4.
    # This gives four sub-problems:
    # 1. 2x2x4 (top-left of A, top-left of B)
    # 2. 2x1x4 (top-right of A, top-left of B)
    # 3. 2x2x4 (top-left of A, bottom-right of B)
    # 4. 2x1x4 (top-right of A, bottom-right of B)
    
    # For 2x2x4: We can use Strassen on 2x2x2 blocks.
    # Split p=4 into two p=2 blocks. Each 2x2x2 uses rank 7.
    # Total rank for 2x2x4 is 7 + 7 = 14.
    
    # For 2x1x4: This is a linear map from 4-dim to 2-dim.
    # It can be decomposed into 4 rank-1 terms (schoolbook).
    # Total rank for 2x1x4 is 4.
    
    # Total rank = (14 + 4) + (14 + 4) = 36.
    # This is better than the previous 44.
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # Indices:
    # U: (i, j) where i in {0,1}, j in {0,1}
    # V: (j, k) where j in {0,1}, k in {0,1}
    # W: (k, i) where k in {0,1}, i in {0,1}
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    # Helper to add schoolbook 2x1x4
    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off)][col] = 1
                V[(j_off)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: p in 0..3
    # Sub-block 1a: 2x2x2 (j=0,1; k=0,1)
    add_strassen_2x2x2(0, 0, 0)
    # Sub-block 1b: 2x2x2 (j=0,1; k=2,3)
    add_strassen_2x2x2(0, 0, 2)
    # Sub-block 1c: 2x1x4 (j=2; k=0,1,2,3)
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: p in 4..7
    # Sub-block 2a: 2x2x2 (j=0,1; k=4,5)
    add_strassen_2x2x2(0, 0, 4)
    # Sub-block 2b: 2x2x2 (j=0,1; k=6,7)
    add_strassen_2x2x2(0, 0, 6)
    # Sub-block 2c: 2x1x4 (j=2; k=4,5,6,7)
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
