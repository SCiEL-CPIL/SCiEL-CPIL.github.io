import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # The total rank is 78.
    # We use a list of tuples (u_idx, val), (v_idx, val), (w_idx, val) for each rank column.
    # Then we assemble U, V, W.
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_trivial_block(i_start, j_start, k_start, ni, nj, nk):
        """
        Adds a trivial decomposition for A[i_start:i_start+ni, j_start:j_start+nj] * B[j_start:j_start+nj, k_start:k_start+nk]
        Note: Inner dimension is nj.
        """
        nonlocal rank
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u_idx = (i_start + i) * m + (j_start + j)
                    v_idx = (j_start + j) * p + (k_start + k)
                    w_idx = (k_start + k) * n + (i_start + i)
                    
                    U[u_idx][rank] = 1
                    V[v_idx][rank] = 1
                    W[w_idx][rank] = 1
                    rank += 1

    def add_strassen_2x2x2(i1, i2, j1, j2, k1, k2):
        """
        Adds Strassen decomposition for 2x2 A block, 2x2 B block, 2x2 C block.
        A rows i1, i2; cols j1, j2
        B rows j1, j2; cols k1, k2
        C rows i1, i2; cols k1, k2
        """
        nonlocal rank
        
        # M1 = (a+d)(e+h)
        # A: (i1,j1), (i2,j2)
        # B: (j1,k1), (j2,k2)
        # C: (i1,k1)+, (i2,k2)+
        U[(i1)*m + j1][rank] = 1; U[(i2)*m + j2][rank] = 1
        V[(j1)*p + k1][rank] = 1; V[(j2)*p + k2][rank] = 1
        W[(k1)*n + i1][rank] = 1; W[(k2)*n + i2][rank] = 1
        rank += 1
        
        # M2 = (c+d)e
        # A: (i2,j1), (i2,j2)
        # B: (j1,k1)
        # C: (i2,k1)+
        U[(i2)*m + j1][rank] = 1; U[(i2)*m + j2][rank] = 1
        V[(j1)*p + k1][rank] = 1
        W[(k1)*n + i2][rank] = 1
        rank += 1
        
        # M3 = a(f-h)
        # A: (i1,j1)
        # B: (j1,k2), (j2,k2)
        # C: (i1,k2)+
        U[(i1)*m + j1][rank] = 1
        V[(j1)*p + k2][rank] = 1; V[(j2)*p + k2][rank] = -1
        W[(k2)*n + i1][rank] = 1
        rank += 1
        
        # M4 = d(g-e)
        # A: (i2,j2)
        # B: (j2,k1), (j1,k1)
        # C: (i1,k1)+, (i2,k1)+
        U[(i2)*m + j2][rank] = 1
        V[(j2)*p + k1][rank] = 1; V[(j1)*p + k1][rank] = -1
        W[(k1)*n + i1][rank] = 1; W[(k1)*n + i2][rank] = 1
        rank += 1
        
        # M5 = (a+b)h
        # A: (i1,j1), (i1,j2)
        # B: (j2,k2)
        # C: (i1,k1)-, (i1,k2)+
        U[(i1)*m + j1][rank] = 1; U[(i1)*m + j2][rank] = 1
        V[(j2)*p + k2][rank] = 1
        W[(k1)*n + i1][rank] = -1; W[(k2)*n + i1][rank] = 1
        rank += 1
        
        # M6 = (c-a)(e+f)
        # A: (i2,j1), (i1,j1)
        # B: (j1,k1), (j1,k2)
        # C: (i2,k2)+
        U[(i2)*m + j1][rank] = 1; U[(i1)*m + j1][rank] = -1
        V[(j1)*p + k1][rank] = 1; V[(j1)*p + k2][rank] = 1
        W[(k2)*n + i2][rank] = 1
        rank += 1
        
        # M7 = (b-d)(g+h)
        # A: (i1,j2), (i2,j2)
        # B: (j2,k1), (j2,k2)
        # C: (i1,k1)+, (i2,k2)-
        U[(i1)*m + j2][rank] = 1; U[(i2)*m + j2][rank] = -1
        V[(j2)*p + k1][rank] = 1; V[(j2)*p + k2][rank] = 1
        W[(k1)*n + i1][rank] = 1; W[(k2)*n + i2][rank] = -1
        rank += 1

    # Decomposition:
    # A is 3x4, B is 4x7.
    # Split inner dim 4 into 2+2.
    # Left part: A_L (3x2, cols 0,1), B_L (2x7, rows 0,1)
    # Right part: A_R (3x2, cols 2,3), B_R (2x7, rows 2,3)
    
    # Process Left Part (A_L B_L)
    # Split A_L (3x2) into A_L_top (2x2, rows 0,1) and A_L_bot (1x2, row 2)
    
    # 1. A_L_bot (1x2) * B_L (2x7) -> Trivial
    # i_start=2, j_start=0, k_start=0, ni=1, nj=2, nk=7
    add_trivial_block(2, 0, 0, 1, 2, 7)
    
    # 2. A_L_top (2x2) * B_L (2x7)
    # Split B_L (2x7) into B_L1 (2x2, k=0,1), B_L2 (2x2, k=2,3), B_L3 (2x2, k=4,5), B_L4 (2x1, k=6)
    
    # A_L_top rows 0,1; cols 0,1
    # B_L1 rows 0,1; cols 0,1
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1)
    
    # B_L2 rows 0,1; cols 2,3
    add_strassen_2x2x2(0, 1, 0, 1, 2, 3)
    
    # B_L3 rows 0,1; cols 4,5
    add_strassen_2x2x2(0, 1, 0, 1, 4, 5)
    
    # B_L4 rows 0,1; cols 6
    # A_L_top (2x2) * B_L4 (2x1) -> Trivial
    # i_start=0, j_start=0, k_start=6, ni=2, nj=2, nk=1
    add_trivial_block(0, 0, 6, 2, 2, 1)
    
    # Process Right Part (A_R B_R)
    # A_R is rows 0,1,2; cols 2,3
    # B_R is rows 2,3; cols 0,6
    
    # 3. A_R_bot (1x2, row 2, cols 2,3) * B_R (2x7) -> Trivial
    # i_start=2, j_start=2, k_start=0, ni=1, nj=2, nk=7
    add_trivial_block(2, 2, 0, 1, 2, 7)
    
    # 4. A_R_top (2x2, rows 0,1; cols 2,3) * B_R (2x7)
    # Split B_R (2x7) into B_R1 (2x2, k=0,1), B_R2 (2x2, k=2,3), B_R3 (2x2, k=4,5), B_R4 (2x1, k=6)
    
    # A_R_top rows 0,1; cols 2,3
    # B_R1 rows 2,3; cols 0,1
    add_strassen_2x2x2(0, 1, 2, 3, 0, 1)
    
    # B_R2 rows 2,3; cols 2,3
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3)
    
    # B_R3 rows 2,3; cols 4,5
    add_strassen_2x2x2(0, 1, 2, 3, 4, 5)
    
    # B_R4 rows 2,3; cols 6
    # A_R_top (2x2) * B_R4 (2x1) -> Trivial
    # i_start=0, j_start=2, k_start=6, ni=2, nj=2, nk=1
    add_trivial_block(0, 2, 6, 2, 2, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
