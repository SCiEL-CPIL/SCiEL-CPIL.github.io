import numpy as np

def construct():
    n, m, p = 3, 4, 8
    rank_total = 88
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_offset, j_offset, k_offset):
        nonlocal col
        # Local indices i' in 0,1; j' in 0,1; k' in 0,1
        # Global i = i_offset + i'
        # Global j = j_offset + j'
        # Global k = k_offset + k'
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        # U: A00(1), A11(1)
        U[i_offset*m + j_offset, col] = 1
        U[(i_offset+1)*m + (j_offset+1), col] = 1
        # V: B00(1), B11(1)
        V[j_offset*p + k_offset, col] = 1
        V[(j_offset+1)*p + (k_offset+1), col] = 1
        # W: C00(1), C11(1)
        W[k_offset*n + i_offset, col] = 1
        W[(k_offset+1)*n + (i_offset+1), col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10(+1), C11(-1)
        # U: A10(1), A11(1)
        U[(i_offset+1)*m + j_offset, col] = 1
        U[(i_offset+1)*m + (j_offset+1), col] = 1
        # V: B00(1)
        V[j_offset*p + k_offset, col] = 1
        # W: C10(1), C11(-1)
        W[k_offset*n + (i_offset+1), col] = 1
        W[(k_offset+1)*n + (i_offset+1), col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01(+1), C11(+1)
        # U: A00(1)
        U[i_offset*m + j_offset, col] = 1
        # V: B01(1), B11(-1)
        V[j_offset*p + (k_offset+1), col] = 1
        V[(j_offset+1)*p + (k_offset+1), col] = -1
        # W: C01(1), C11(1)
        W[(k_offset+1)*n + i_offset, col] = 1
        W[(k_offset+1)*n + (i_offset+1), col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00(+1), C10(+1)
        # U: A11(1)
        U[(i_offset+1)*m + (j_offset+1), col] = 1
        # V: B10(1), B00(-1)
        V[(j_offset+1)*p + k_offset, col] = 1
        V[j_offset*p + k_offset, col] = -1
        # W: C00(1), C10(1)
        W[k_offset*n + i_offset, col] = 1
        W[k_offset*n + (i_offset+1), col] = 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00(-1), C01(+1)
        # U: A00(1), A01(1)
        U[i_offset*m + j_offset, col] = 1
        U[i_offset*m + (j_offset+1), col] = 1
        # V: B11(1)
        V[(j_offset+1)*p + (k_offset+1), col] = 1
        # W: C00(-1), C01(1)
        W[k_offset*n + i_offset, col] = -1
        W[(k_offset+1)*n + i_offset, col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11(+1)
        # U: A10(1), A00(-1)
        U[(i_offset+1)*m + j_offset, col] = 1
        U[i_offset*m + j_offset, col] = -1
        # V: B00(1), B01(1)
        V[j_offset*p + k_offset, col] = 1
        V[j_offset*p + (k_offset+1), col] = 1
        # W: C11(1)
        W[(k_offset+1)*n + (i_offset+1), col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00(+1)
        # U: A01(1), A11(-1)
        U[i_offset*m + (j_offset+1), col] = 1
        U[(i_offset+1)*m + (j_offset+1), col] = -1
        # V: B10(1), B11(1)
        V[(j_offset+1)*p + k_offset, col] = 1
        V[(j_offset+1)*p + (k_offset+1), col] = 1
        # W: C00(1)
        W[k_offset*n + i_offset, col] = 1
        col += 1

    # Helper to add a 1x2x2 schoolbook block
    def add_schoolbook_122(i_idx, j_offset, k_offset):
        nonlocal col
        # A is 1x2: A0, A1
        # B is 2x2: B00, B01, B10, B11
        # C is 1x2: C0, C1
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        
        # Term 1: A0*B00 -> C0
        U[i_idx*m + j_offset, col] = 1
        V[j_offset*p + k_offset, col] = 1
        W[k_offset*n + i_idx, col] = 1
        col += 1
        
        # Term 2: A1*B10 -> C0
        U[i_idx*m + (j_offset+1), col] = 1
        V[(j_offset+1)*p + k_offset, col] = 1
        W[k_offset*n + i_idx, col] = 1
        col += 1
        
        # Term 3: A0*B01 -> C1
        U[i_idx*m + j_offset, col] = 1
        V[j_offset*p + (k_offset+1), col] = 1
        W[(k_offset+1)*n + i_idx, col] = 1
        col += 1
        
        # Term 4: A1*B11 -> C1
        U[i_idx*m + (j_offset+1), col] = 1
        V[(j_offset+1)*p + (k_offset+1), col] = 1
        W[(k_offset+1)*n + i_idx, col] = 1
        col += 1

    # Block 1: i in {0,1}
    # Sub-block 1a: j in {0,1}
    for k_block in range(4):
        k_offset = k_block * 2
        add_strassen_222(0, 0, k_offset)
        
    # Sub-block 1b: j in {2,3}
    for k_block in range(4):
        k_offset = k_block * 2
        add_strassen_222(0, 2, k_offset)

    # Block 2: i in {2}
    # Sub-block 2a: j in {0,1}
    for k_block in range(4):
        k_offset = k_block * 2
        add_schoolbook_122(2, 0, k_offset)
        
    # Sub-block 2b: j in {2,3}
    for k_block in range(4):
        k_offset = k_block * 2
        add_schoolbook_122(2, 2, k_offset)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
