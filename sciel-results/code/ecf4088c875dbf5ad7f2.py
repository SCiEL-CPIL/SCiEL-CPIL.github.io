import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 28.
    # Split m=3 into m1=1, m2=2.
    # Split p=5 into p1=2, p2=3.
    # Sub-problems:
    # 1. (n=2, m1=1, p1=2) -> Rank 4
    # 2. (n=2, m1=1, p2=3) -> Rank 6
    # 3. (n=2, m2=2, p1=2) -> Rank 7 (Strassen)
    # 4. (n=2, m2=2, p2=3) -> Split p2 into 1 and 2.
    #    4a. (n=2, m2=2, p2a=1) -> Rank 4
    #    4b. (n=2, m2=2, p2b=2) -> Rank 7 (Strassen)
    # Total Rank = 4 + 6 + 7 + 4 + 7 = 28.

    total_rank = 28
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0

    # Helper to add a schoolbook sub-decomposition
    def add_schoolbook(i_start, j_start, k_start, ni, nj, np, u, v, w, offset):
        rank_sub = ni * nj * np
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(np):
                    i = i_start + ii
                    j = j_start + jj
                    k = k_start + kk
                    
                    u[i * m + j][offset] = 1
                    v[j * p + k][offset] = 1
                    w[k * n + i][offset] = 1
                    offset += 1
        return offset

    # Helper to add a Strassen sub-decomposition for 2x2x2
    # i_start, j_start, k_start are the global indices of the top-left corner
    def add_strassen(i_start, j_start, k_start, u, v, w, offset):
        # Strassen formulas for 2x2x2
        # A is 2x2, B is 2x2, C is 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We define the contributions to U, V, W for each of the 7 multiplications
        # Each multiplication is a product of a linear combination of A and a linear combination of B.
        # U stores coefficients for A, V for B, W for C.
        
        # M1
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        u[i_start * m + j_start][offset] = 1
        u[(i_start+1) * m + (j_start+1)][offset] = 1
        v[j_start * p + k_start][offset] = 1
        v[(j_start+1) * p + (k_start+1)][offset] = 1
        w[k_start * n + i_start][offset] = 1
        w[(k_start+1) * n + (i_start+1)][offset] = 1
        offset += 1
        
        # M2
        # U: A10(1), A11(1)
        # V: B00(1)
        # W: C10(1)
        u[(i_start+1) * m + j_start][offset] = 1
        u[(i_start+1) * m + (j_start+1)][offset] = 1
        v[j_start * p + k_start][offset] = 1
        w[k_start * n + (i_start+1)][offset] = 1
        offset += 1
        
        # M3
        # U: A00(1)
        # V: B01(1), B11(-1)
        # W: C01(1)
        u[i_start * m + j_start][offset] = 1
        v[j_start * p + (k_start+1)][offset] = 1
        v[(j_start+1) * p + (k_start+1)][offset] = -1
        w[(k_start+1) * n + i_start][offset] = 1
        offset += 1
        
        # M4
        # U: A11(1)
        # V: B10(1), B00(-1)
        # W: C00(1), C10(1)
        u[(i_start+1) * m + (j_start+1)][offset] = 1
        v[(j_start+1) * p + k_start][offset] = 1
        v[j_start * p + k_start][offset] = -1
        w[k_start * n + i_start][offset] = 1
        w[k_start * n + (i_start+1)][offset] = 1
        offset += 1
        
        # M5
        # U: A00(1), A01(1)
        # V: B11(1)
        # W: C00(-1), C11(1)
        u[i_start * m + j_start][offset] = 1
        u[i_start * m + (j_start+1)][offset] = 1
        v[(j_start+1) * p + (k_start+1)][offset] = 1
        w[k_start * n + i_start][offset] = -1
        w[(k_start+1) * n + (i_start+1)][offset] = 1
        offset += 1
        
        # M6
        # U: A10(1), A00(-1)
        # V: B00(1), B01(1)
        # W: C11(1)
        u[(i_start+1) * m + j_start][offset] = 1
        u[i_start * m + j_start][offset] = -1
        v[j_start * p + k_start][offset] = 1
        v[j_start * p + (k_start+1)][offset] = 1
        w[(k_start+1) * n + (i_start+1)][offset] = 1
        offset += 1
        
        # M7
        # U: A01(1), A11(-1)
        # V: B10(1), B11(1)
        # W: C00(1), C11(-1)
        u[i_start * m + (j_start+1)][offset] = 1
        u[(i_start+1) * m + (j_start+1)][offset] = -1
        v[(j_start+1) * p + k_start][offset] = 1
        v[(j_start+1) * p + (k_start+1)][offset] = 1
        w[k_start * n + i_start][offset] = 1
        w[(k_start+1) * n + (i_start+1)][offset] = -1
        offset += 1
        
        return offset

    # 1. i in [0,1], j=0, k in [0,1]
    # Schoolbook 2x1x2
    col_offset = add_schoolbook(0, 0, 0, 2, 1, 2, U, V, W, col_offset)
    
    # 2. i in [0,1], j=0, k in [2,4]
    # Schoolbook 2x1x3
    col_offset = add_schoolbook(0, 0, 2, 2, 1, 3, U, V, W, col_offset)
    
    # 3. i in [0,1], j in [1,2], k in [0,1]
    # Strassen 2x2x2
    col_offset = add_strassen(0, 1, 0, U, V, W, col_offset)
    
    # 4a. i in [0,1], j in [1,2], k=2
    # Schoolbook 2x2x1
    col_offset = add_schoolbook(0, 1, 2, 2, 2, 1, U, V, W, col_offset)
    
    # 4b. i in [0,1], j in [1,2], k in [3,4]
    # Strassen 2x2x2
    col_offset = add_strassen(0, 1, 3, U, V, W, col_offset)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
