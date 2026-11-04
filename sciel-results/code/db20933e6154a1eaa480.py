"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    # U: (n*m, rank) -> 6 x 55
    # V: (m*p, rank) -> 30 x 55
    # W: (p*n, rank) -> 20 x 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices 0,1 map to global i_off+0, i_off+1 etc.
        # U indices: (i_off+li)*m + (j_off+lj)
        # V indices: (j_off+lj)*p + (k_off+lk)
        # W indices: (k_off+lk)*n + (i_off+li)
        
        # Strassen matrices M1..M7
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Coefficients for U (A part)
        # Row index in U: i*m + j
        # We add 1 or -1 to specific (i,j) pairs for each M.
        
        # M1
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        # C00 = M1 + ... -> W[k0, i0] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        # C11 = M1 + ... -> W[k1, i1] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        # C10 = M2 + ... -> W[k0, i1] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        # C11 = ... - M2 ... -> W[k1, i1] += -1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # M3
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        # C01 = M3 + ... -> W[k1, i0] += 1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        # C11 = ... + M3 ... -> W[k1, i1] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M4
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        # C00 = ... + M4 ... -> W[k0, i0] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        # C10 = ... + M4 ... -> W[k0, i1] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # M5
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        # C00 = ... - M5 ... -> W[k0, i0] += -1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        # C01 = ... + M5 ... -> W[k1, i0] += 1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # M6
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        # C11 = ... + M6 ... -> W[k1, i1] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M7
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        # C00 = ... + M7 ... -> W[k0, i0] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_schoolbook_1x1(i_off, j_idx, k_start, k_end):
        nonlocal col
        # i_off is 0, so i is 0, 1
        # j_idx is fixed (2)
        # k varies
        for i in range(n):
            for k in range(k_start, k_end):
                U[(i_off+i)*m + j_idx][col] += 1
                V[j_idx*p + k][col] += 1
                W[k*n + (i_off+i)][col] += 1
                col += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    
    # Block 3: j=2, k in {0,1,2,3}
    add_schoolbook_1x1(0, 2, 0, 4)
    
    # Block 4: j in {0,1}, k in {4,5}
    add_strassen(0, 0, 4)
    
    # Block 5: j in {0,1}, k in {6,7}
    add_strassen(0, 0, 6)
    
    # Block 6: j in {0,1}, k in {8,9}
    add_strassen(0, 0, 8)
    
    # Block 7: j=2, k in {4,5,6,7,8,9}
    add_schoolbook_1x1(0, 2, 4, 10)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
