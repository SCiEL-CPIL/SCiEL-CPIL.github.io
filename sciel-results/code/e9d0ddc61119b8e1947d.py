import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (or low-rank) exact tensor decomposition for 4x4x5 matrix multiplication.
    Uses a deterministic Strassen-based block construction to ensure speed and validity.
    """
    n, m, p = DIMENSIONS
    
    # We construct the decomposition using a block approach.
    # A is 4x4, B is 4x5.
    # Split A into 2x2 blocks of size 2x2.
    # Split B into 2x2 blocks: top-left (2x2), top-right (2x3), bottom-left (2x2), bottom-right (2x3).
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def get_u_idx(i, j):
        return i * m + j
        
    def get_v_idx(j, k):
        return j * p + k
        
    def get_w_idx(k, i):
        return k * n + i

    # Strassen for 2x2x2 multiplication
    # Inputs: A block at (ai, aj), B block at (bj, bk) (both 2x2)
    # Output: C block at (bk, ai) is not quite right, C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # If A is block (ai, aj) and B is block (bj, bk), they contribute to C block (ai, bk)?
    # No, standard block multiplication: C_{block_i, block_k} = sum_j A_{block_i, block_j} B_{block_j, block_k}.
    # Here blocks are 2x2.
    # A indices: 0,1 (rows), 0,1 (cols) -> Block (0,0), (0,1), (1,0), (1,1)
    # B indices: 0,1 (rows), 0,1,2,3 (cols) -> Block (0,0) [cols 0,1], (0,1) [cols 2,3,4]? No, B is 4x5.
    # Let's stick to the previous helper functions logic which was correct for the indices.
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        # ai, aj: block row/col of A (0 or 1). Actual rows ai*2, ai*2+1; cols aj*2, aj*2+1.
        # bj, bk: block row/col of B (0 or 1). Actual rows bj*2, bj*2+1; cols bk*2, bk*2+1.
        # This computes contribution to C block (ai, bk) from A(ai,aj) and B(bj,bk).
        # Wait, C_{i,k} depends on A_{i,j} and B_{j,k}.
        # So A block (ai, aj) and B block (aj, bk) contribute to C block (ai, bk).
        # The previous code called add_strassen_2x2_2(0, 0, 0, 0) for C00.
        # A(0,0) * B(0,0) -> C(0,0). Correct.
        # A(0,1) * B(1,0) -> C(0,0). Correct.
        
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai*2, aj*2)] = 1; u[get_u_idx(ai*2+1, aj*2+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj*2, bk*2)] = 1; v[get_v_idx(bj*2+1, bk*2+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk*2, ai*2)] = 1; w[get_w_idx(bk*2+1, ai*2+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai*2+1, aj*2)] = 1; u[get_u_idx(ai*2+1, aj*2+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj*2, bk*2)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk*2, ai*2+1)] = 1; w[get_w_idx(bk*2+1, ai*2+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai*2, aj*2)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj*2, bk*2+1)] = 1; v[get_v_idx(bj*2+1, bk*2+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk*2+1, ai*2)] = 1; w[get_w_idx(bk*2+1, ai*2+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai*2+1, aj*2+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj*2+1, bk*2)] = 1; v[get_v_idx(bj*2, bk*2)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk*2, ai*2)] = 1; w[get_w_idx(bk*2, ai*2+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai*2, aj*2)] = 1; u[get_u_idx(ai*2, aj*2+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj*2+1, bk*2+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk*2, ai*2)] = -1; w[get_w_idx(bk*2+1, ai*2)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai*2+1, aj*2)] = 1; u[get_u_idx(ai*2, aj*2)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj*2, bk*2)] = 1; v[get_v_idx(bj*2, bk*2+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk*2+1, ai*2+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai*2, aj*2+1)] = 1; u[get_u_idx(ai*2+1, aj*2+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj*2+1, bk*2)] = 1; v[get_v_idx(bj*2+1, bk*2+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk*2, ai*2)] = 1
        add_term(u, v, w)

    # Helper for 2x2 A block and 2x3 B block
    # A block (ai, aj) is 2x2.
    # B block (bj, bk) covers rows bj*2, bj*2+1 and cols bk*2, bk*2+1, bk*2+2.
    def add_2x2x3(ai, aj, bj, bk):
        # Use Strassen for the first 2 columns of B block
        add_strassen_2x2_2(ai, aj, bj, bk)
        # Naive for the 3rd column (col index bk*2+2)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai*2+di, aj*2+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj*2+dj, bk*2+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk*2+2, ai*2+di)] = 1
                add_term(u, v, w)

    # C00 (rows 0-1, cols 0-1) = A00 B00 + A01 B10
    # A00: ai=0, aj=0. B00: bj=0, bk=0.
    add_strassen_2x2_2(0, 0, 0, 0)
    # A01: ai=0, aj=1. B10: bj=1, bk=0.
    add_strassen_2x2_2(0, 1, 1, 0)
    
    # C01 (rows 0-1, cols 2-4) = A00 B01 + A01 B11
    # B01 is 2x3 (rows 0-1, cols 2-4). bj=0, bk=0 (for cols 2,3,4? No, bk is block index.
    # In our scheme, B block (0,0) is cols 0,1. B block (0,1) is cols 2,3,4?
    # Yes, we defined B blocks as 2x2 and 2x3.
    # So for C01, we use B block (0,1) which is 2x3.
    add_2x2x3(0, 0, 0, 1) # A00 * B01
    add_2x2x3(0, 1, 1, 1) # A01 * B11
    
    # C10 (rows 2-3, cols 0-1) = A10 B00 + A11 B10
    add_strassen_2x2_2(1, 0, 0, 0) # A10 * B00
    add_strassen_2x2_2(1, 1, 1, 0) # A11 * B10
    
    # C11 (rows 2-3, cols 2-4) = A10 B01 + A11 B11
    add_2x2x3(1, 0, 0, 1) # A10 * B01
    add_2x2x3(1, 1, 1, 1) # A11 * B11

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
