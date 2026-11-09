import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(a_idx, b_idx, c_idx, u_scale=1.0, v_scale=1.0, w_scale=1.0):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(n * p)
        u[a_idx] = u_scale
        v[b_idx] = v_scale
        w[c_idx] = w_scale
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_2x2x2(i0, j0, k0):
        # Strassen's algorithm for 2x2x2 matrix multiplication
        # A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] = C[k0:k0+2, i0:i0+2]
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # M1
        # U: A00 + A11, V: B00 + B11, W: C00 + C11
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0) * m + j0] = 1.0
        u[(i0+1) * m + j0+1] = 1.0
        v[j0 * p + k0] = 1.0
        v[(j0+1) * p + k0+1] = 1.0
        w[k0 * n + i0] = 1.0
        w[(k0+1) * n + i0+1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M2
        # U: A10 + A11, V: B00, W: C10 + C11
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0+1) * m + j0] = 1.0
        u[(i0+1) * m + j0+1] = 1.0
        v[j0 * p + k0] = 1.0
        w[(k0+1) * n + i0] = 1.0
        w[(k0+1) * n + i0+1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M3
        # U: A00, V: B01 - B11, W: C01 + C11
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0) * m + j0] = 1.0
        v[j0 * p + k0+1] = 1.0
        v[(j0+1) * p + k0+1] = -1.0
        w[k0 * n + i0+1] = 1.0
        w[(k0+1) * n + i0+1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M4
        # U: A11, V: B10 - B00, W: C00 + C10
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0+1) * m + j0+1] = 1.0
        v[(j0+1) * p + k0] = 1.0
        v[j0 * p + k0] = -1.0
        w[k0 * n + i0] = 1.0
        w[(k0+1) * n + i0] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M5
        # U: A00 + A01, V: B11, W: C01 - C00
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0) * m + j0] = 1.0
        u[(i0) * m + j0+1] = 1.0
        v[(j0+1) * p + k0+1] = 1.0
        w[k0 * n + i0+1] = 1.0
        w[k0 * n + i0] = -1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M6
        # U: A10 - A00, V: B00 + B01, W: C11
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0+1) * m + j0] = 1.0
        u[(i0) * m + j0] = -1.0
        v[j0 * p + k0] = 1.0
        v[j0 * p + k0+1] = 1.0
        w[(k0+1) * n + i0+1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M7
        # U: A01 - A11, V: B10 + B11, W: C00
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(n * p)
        u[(i0) * m + j0+1] = 1.0
        u[(i0+1) * m + j0+1] = -1.0
        v[(j0+1) * p + k0] = 1.0
        v[(j0+1) * p + k0+1] = 1.0
        w[k0 * n + i0] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    def add_schoolbook_block(n_b, m_b, p_b, i_off, j_off, k_off):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    a_idx = (i_off + i) * m + (j_off + j)
                    b_idx = (j_off + j) * p + (k_off + k)
                    c_idx = (k_off + k) * n + (i_off + i)
                    add_rank1(a_idx, b_idx, c_idx)
    
    # Decompose 4x5x5 into blocks:
    # Split m=5 into 2+3, p=5 into 2+3
    # Block 1: 4x2x2 -> use Strassen on each 2x2x2 subblock (2 blocks of 2x2x2)
    # Block 2: 4x2x3 -> schoolbook
    # Block 3: 4x3x2 -> schoolbook
    # Block 4: 4x3x3 -> schoolbook
    
    # Block 1: 4x2x2, split into two 2x2x2 blocks
    add_strassen_2x2x2(0, 0, 0)  # rows 0-1
    add_strassen_2x2x2(2, 0, 0)  # rows 2-3
    
    # Block 2: 4x2x3, schoolbook
    add_schoolbook_block(4, 2, 3, 0, 0, 2)
    
    # Block 3: 4x3x2, schoolbook
    add_schoolbook_block(4, 3, 2, 0, 2, 0)
    
    # Block 4: 4x3x3, schoolbook
    add_schoolbook_block(4, 3, 3, 0, 2, 2)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
