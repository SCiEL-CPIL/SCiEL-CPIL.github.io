"""Matrix multiplication 2,4,5 - Block-Strassen decomposition (rank 36)."""
DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    terms = []
    
    def strassen_2x2x2(A_c0, A_c1, B_r0, B_r1, B_d0, B_d1, C_k0, C_k1):
        """
        Computes 2x2 C block at (rows 0,1; cols C_k0,C_k1) from
        A block at (rows 0,1; cols A_c0,A_c1) times B block at (rows B_r0,B_r1; cols B_d0,B_d1)
        
        U indices: A[i,j] -> i*m+j = i*4+j
        V indices: B[j,k] -> j*p+k = j*5+k  
        W indices: C[k,i] -> k*n+i = k*2+i
        """
        ua11 = 0 * 4 + A_c0
        ua12 = 0 * 4 + A_c1
        ua21 = 1 * 4 + A_c0
        ua22 = 1 * 4 + A_c1
        
        vb11 = B_r0 * 5 + B_d0
        vb12 = B_r0 * 5 + B_d1
        vb21 = B_r1 * 5 + B_d0
        vb22 = B_r1 * 5 + B_d1
        
        wc11 = C_k0 * 2 + 0
        wc12 = C_k1 * 2 + 0
        wc21 = C_k0 * 2 + 1
        wc22 = C_k1 * 2 + 1
        
        # M1 = (A11+A22)(B11+B22): C11 += M1, C22 += M1
        terms.append((ua11, 1, ua22, 1, vb11, 1, vb22, 1, wc11, 1, wc22, 1))
        # M2 = (A21+A22)B11: C21 += M2, C22 -= M2
        terms.append((ua21, 1, ua22, 1, vb11, 1, vb22, 0, wc21, 1, wc22, -1))
        # M3 = A11(B12-B22): C12 += M3, C22 += M3
        terms.append((ua11, 1, ua22, 0, vb12, 1, vb22, -1, wc12, 1, wc22, 1))
        # M4 = A22(B21-B11): C11 += M4, C21 += M4
        terms.append((ua11, 0, ua22, 1, vb11, -1, vb21, 1, wc11, 1, wc21, 1))
        # M5 = (A11+A12)B22: C11 -= M5, C12 += M5
        terms.append((ua11, 1, ua12, 1, vb22, 1, vb11, 0, wc11, -1, wc12, 1))
        # M6 = (A21-A11)(B11+B12): C22 += M6
        terms.append((ua21, 1, ua11, -1, vb11, 1, vb12, 1, wc22, 1, wc11, 0))
        # M7 = (A12-A22)(B21+B22): C11 += M7
        terms.append((ua12, 1, ua22, -1, vb21, 1, vb22, 1, wc11, 1, wc22, 0))
    
    # C[:, 0:2] from A_left * B_top[0:2, 0:2] + A_right * B_bottom[2:4, 0:2]
    strassen_2x2x2(A_c0=0, A_c1=1, B_r0=0, B_r1=1, B_d0=0, B_d1=1, C_k0=0, C_k1=1)
    strassen_2x2x2(A_c0=2, A_c1=3, B_r0=2, B_r1=3, B_d0=0, B_d1=1, C_k0=0, C_k1=1)
    
    # C[:, 2:4] from A_left * B_top[0:2, 2:4] + A_right * B_bottom[2:4, 2:4]
    strassen_2x2x2(A_c0=0, A_c1=1, B_r0=0, B_r1=1, B_d0=2, B_d1=3, C_k0=2, C_k1=3)
    strassen_2x2x2(A_c0=2, A_c1=3, B_r0=2, B_r1=3, B_d0=2, B_d1=3, C_k0=2, C_k1=3)
    
    # C[:, 4] from A_left * B_top[0:2, 4] + A_right * B_bottom[2:4, 4] (schoolbook, 8 terms)
    for i in range(2):
        for j in range(2):
            terms.append((i*4+j, 1, j*5+4, 1, 4*2+i, 1))
        for j in range(2, 4):
            terms.append((i*4+j, 1, j*5+4, 1, 4*2+i, 1))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(10)]
    
    for c, term in enumerate(terms):
        u_row1, u_val1, u_row2, u_val2, v_row1, v_val1, v_row2, v_val2, w_row1, w_val1, w_row2, w_val2 = term
        if u_val1 != 0:
            U[u_row1][c] += u_val1
        if u_val2 != 0:
            U[u_row2][c] += u_val2
        if v_val1 != 0:
            V[v_row1][c] += v_val1
        if v_val2 != 0:
            V[v_row2][c] += v_val2
        if w_val1 != 0:
            W[w_row1][c] += w_val1
        if w_val2 != 0:
            W[w_row2][c] += w_val2
    
    # Verify
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(2):
        for k in range(5):
            expected = float(m)  # sum over j of 1*1 = m = 4
            actual = 0.0
            for j in range(4):
                for c in range(rank):
                    actual += U_arr[i*4+j, c] * V_arr[j*5+k, c] * W_arr[k*2+i, c]
            if abs(actual - expected) > 1e-9:
                raise ValueError(f"Verification failed at i={i}, k={k}: expected={expected}, actual={actual}")
    
    return {"U": U, "V": V, "W": W}
