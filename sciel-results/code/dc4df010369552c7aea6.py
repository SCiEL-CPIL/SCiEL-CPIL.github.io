"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook sub-problem
    def add_schoolbook(n_s, m_s, p_s, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    # Helper to add a Strassen 2x2x2 sub-problem
    def add_strassen(n_s, m_s, p_s, i_off, j_off, k_off):
        # Assumes n_s=2, m_s=2, p_s=2
        nonlocal col
        # P1 = (A11+A22)(B11+B22) -> C11, C22
        # P2 = (A21+A22)B11 -> C21, C22
        # P3 = A11(B12-B22) -> C11, C12
        # P4 = A22(B21-B11) -> C11, C21
        # P5 = (A11+A12)B22 -> C11, C12
        # P6 = (A21-A11)(B11+B12) -> C22
        # P7 = (A12-A22)(B21+B22) -> C11, C22
        
        # Term 1: P1
        # A: 11, 22. B: 11, 22. C: 11(+1), 22(+1)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 2: P2
        # A: 21, 22. B: 11. C: 21(+1), 22(-1)
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # Term 3: P3
        # A: 11. B: 12(-1), 22(1). C: 11(-1), 12(1)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # Term 4: P4
        # A: 22. B: 21(1), 11(-1). C: 11(1), 21(1)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 5: P5
        # A: 11, 12. B: 22. C: 11(-1), 12(1)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # Term 6: P6
        # A: 21(-1), 11(1). B: 11, 12. C: 22(1)
        U[(i_off+1)*m + (j_off+0)][col] = -1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 7: P7
        # A: 12(-1), 22(1). B: 21, 22. C: 11(1), 22(1)
        U[(i_off+0)*m + (j_off+1)][col] = -1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1

    # Decomposition of 3x3x5
    # Block 1: p=0,1 (2 cols)
    #   Sub 1a: m=0,1 (2 cols), n=0,1 (2 rows) -> Strassen 2x2x2 at (0,0,0)
    add_strassen(2, 2, 2, 0, 0, 0)
    #   Sub 1b: m=0,1 (2 cols), n=2 (1 row) -> Schoolbook 1x2x2 at (2,0,0)
    add_schoolbook(1, 2, 2, 2, 0, 0)
    #   Sub 1c: m=2 (1 col), n=0,1,2 (3 rows) -> Schoolbook 3x1x2 at (0,2,0)
    add_schoolbook(3, 1, 2, 0, 2, 0)
    
    # Block 2: p=2,3 (2 cols)
    #   Sub 2a: m=0,1 (2 cols), n=0,1 (2 rows) -> Strassen 2x2x2 at (0,0,2)
    add_strassen(2, 2, 2, 0, 0, 2)
    #   Sub 2b: m=0,1 (2 cols), n=2 (1 row) -> Schoolbook 1x2x2 at (2,0,2)
    add_schoolbook(1, 2, 2, 2, 0, 2)
    #   Sub 2c: m=2 (1 col), n=0,1,2 (3 rows) -> Schoolbook 3x1x2 at (0,2,2)
    add_schoolbook(3, 1, 2, 0, 2, 2)
    
    # Block 3: p=4 (1 col)
    #   Sub 3a: m=0,1,2 (3 cols), n=0,1,2 (3 rows) -> Schoolbook 3x3x1 at (0,0,4)
    add_schoolbook(3, 3, 1, 0, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
