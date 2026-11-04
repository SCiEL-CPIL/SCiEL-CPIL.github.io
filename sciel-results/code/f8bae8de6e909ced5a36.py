import numpy as np
import time

def construct():
    deadline = time.time() + 110
    n, m, p = 3, 5, 5
    
    # Total rank
    rank = 71
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_schoolbook(n_s, m_s, p_s, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[(i_off + i) * m + (j_off + j), col] = 1
                    V[(j_off + j) * p + (k_off + k), col] = 1
                    W[(k_off + k) * n + (i_off + i), col] = 1
                    col += 1

    def add_strassen(n_s, m_s, p_s, i_off, j_off, k_off):
        # Only for 2,2,2
        nonlocal col
        # Strassen decomposition for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Factors for U (A part), V (B part), W (C part)
        # U: (2*2, 7)
        # V: (2*2, 7)
        # W: (2*2, 7)
        
        # Indices in local block:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # Term 1: (A00+A11)(B00+B11)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = -1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        col += 1
        
        # Term 3: A00(B10-B11)
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 4: A11(B01-B00)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B10)
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B01+B11)
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

    # Part 1: n=2 (rows 0,1)
    # m split: 2, 2, 1
    # p split: 2, 2, 1
    
    # j=0 (m 0-1), k=0 (p 0-1): 2x2x2 Strassen
    add_strassen(2, 2, 2, 0, 0, 0)
    # j=0 (m 0-1), k=1 (p 2-3): 2x2x2 Strassen
    add_strassen(2, 2, 2, 0, 0, 2)
    # j=0 (m 0-1), k=2 (p 4): 2x2x1 Schoolbook
    add_schoolbook(2, 2, 1, 0, 0, 4)
    
    # j=1 (m 2-3), k=0 (p 0-1): 2x2x2 Strassen
    add_strassen(2, 2, 2, 0, 2, 0)
    # j=1 (m 2-3), k=1 (p 2-3): 2x2x2 Strassen
    add_strassen(2, 2, 2, 0, 2, 2)
    # j=1 (m 2-3), k=2 (p 4): 2x2x1 Schoolbook
    add_schoolbook(2, 2, 1, 0, 2, 4)
    
    # j=2 (m 4), k=0 (p 0-1): 2x1x2 Schoolbook
    add_schoolbook(2, 1, 2, 0, 4, 0)
    # j=2 (m 4), k=1 (p 2-3): 2x1x2 Schoolbook
    add_schoolbook(2, 1, 2, 0, 4, 2)
    # j=2 (m 4), k=2 (p 4): 2x1x1 Schoolbook
    add_schoolbook(2, 1, 1, 0, 4, 4)
    
    # Part 2: n=1 (row 2)
    # j=0 (m 0-1), k=0 (p 0-1): 1x2x2 Schoolbook
    add_schoolbook(1, 2, 2, 2, 0, 0)
    # j=0 (m 0-1), k=1 (p 2-3): 1x2x2 Schoolbook
    add_schoolbook(1, 2, 2, 2, 0, 2)
    # j=0 (m 0-1), k=2 (p 4): 1x2x1 Schoolbook
    add_schoolbook(1, 2, 1, 2, 0, 4)
    
    # j=1 (m 2-3), k=0 (p 0-1): 1x2x2 Schoolbook
    add_schoolbook(1, 2, 2, 2, 2, 0)
    # j=1 (m 2-3), k=1 (p 2-3): 1x2x2 Schoolbook
    add_schoolbook(1, 2, 2, 2, 2, 2)
    # j=1 (m 2-3), k=2 (p 4): 1x2x1 Schoolbook
    add_schoolbook(1, 2, 1, 2, 2, 4)
    
    # j=2 (m 4), k=0 (p 0-1): 1x1x2 Schoolbook
    add_schoolbook(1, 1, 2, 2, 4, 0)
    # j=2 (m 4), k=1 (p 2-3): 1x1x2 Schoolbook
    add_schoolbook(1, 1, 2, 2, 4, 2)
    # j=2 (m 4), k=2 (p 4): 1x1x1 Schoolbook
    add_schoolbook(1, 1, 1, 2, 4, 4)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
