import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 145.0

    # Target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.max(np.abs(T_approx - T)) < tol

    # Schoolbook
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, r] = 1.0
                V_sb[b, r] = 1.0
                W_sb[c, r] = 1.0
                r += 1
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Strassen 2x2x2
    def strassen_2x2(A, B):
        # Returns 7 linear forms
        # A, B are 2x2
        M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        M2 = (A[1,0] + A[1,1]) * B[0,0]
        M3 = A[0,0] * (B[0,1] - B[1,1])
        M4 = A[1,1] * (B[1,0] - B[0,0])
        M5 = (A[0,0] + A[0,1]) * B[1,1]
        M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        return [M1, M2, M3, M4, M5, M6, M7]

    # Construct Block Solution (Rank ~40-56)
    # This is a heuristic construction
    U_block = []
    V_block = []
    W_block = []
    
    # Top-Left: 2x4 * 4x4 -> 2x4
    # Split A_top (2x4) into A1 (2x2), A2 (2x2)
    # Split B_left (4x4) into B11, B12, B21, B22 (2x2)
    # C_TL = A1 B11 + A2 B21 (for left 2 cols) + A1 B12 + A2 B22 (for right 2 cols)
    # Actually C_TL = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    
    # We will add 7 terms for each 2x2 multiplication
    # 4 multiplications * 7 = 28 terms
    
    # Helper to add Strassen terms for a specific block
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols: indices in A
        # b_rows, b_cols: indices in B
        # c_rows, c_cols: indices in C
        # This is a 2x2x2 block
        
        # Local indices 0,1
        for r in range(7):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Strassen coefficients for U (A)
            # M1: (A00+A11)(B00+B11) -> A00, A11
            # M2: (A10+A11)B00 -> A10, A11
            # M3: A00(B01-B11) -> A00
            # M4: A11(B10-B00) -> A11
            # M5: (A00+A01)B11 -> A00, A01
            # M6: (A10-A00)(B00+B01) -> A10, A00
            # M7: (A01-A11)(B10+B11) -> A01, A11
            
            u_coeffs = [
                [1, 1, 0, 0, 1, -1, 0], # A00
                [1, 1, 0, 0, 0, 1, -1], # A10
                [0, 0, 1, 0, 1, 0, 1],  # A01
                [1, 1, 0, 1, 0, 0, 1]   # A11
            ]
            # Wait, Strassen formulas:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # U coeffs (A)
            # M1: A00(1), A11(1)
            # M2: A10(1), A11(1)
            # M3: A00(1)
            # M4: A11(1)
            # M5: A00(1), A01(1)
            # M6: A10(1), A00(-1)
            # M7: A01(1), A11(-1)
            
            u_vals = [
                [1, 1, 1, 0, 1, -1, 0], # A00
                [0, 1, 0, 0, 0, 1, 1],  # A10
                [0, 0, 0, 0, 1, 0, 1],  # A01
                [1, 1, 0, 1, 0, 0, -1]  # A11
            ]
            
            # V coeffs (B)
            # M1: B00(1), B11(1)
            # M2: B00(1)
            # M3: B01(1), B11(-1)
            # M4: B10(1), B00(-1)
            # M5: B11(1)
            # M6: B00(1), B01(1)
            # M7: B10(1), B11(1)
            
            v_vals = [
                [1, 1, 0, -1, 0, 1, 0], # B00
                [0, 0, 1, 0, 0, 1, 1],  # B01
                [0, 0, 0, 1, 0, 0, 1],  # B10
                [1, 0, -1, 0, 1, 0, 1]  # B11
            ]
            
            # W coeffs (C)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            w_vals = [
                [1, 0, 0, 1, -1, 0, 1], # C00
                [0, 0, 1, 0, 1, 0, 0],  # C01
                [0, 1, 0, 1, 0, 0, 0],  # C10
                [1, -1, 1, 0, 0, 1, 0]  # C11
            ]
            
            # Map local indices to global
            # A local (0,0) -> global (a_rows[0], a_cols[0])
            # A local (1,0) -> global (a_rows[1], a_cols[0])
            # A local (0,1) -> global (a_rows[0], a_cols[1])
            # A local (1,1) -> global (a_rows[1], a_cols[1])
            
            for li in range(2):
                for lj in range(2):
                    gi = a_rows[li]
                    gj = a_cols[lj]
                    u[gi * m + gj] = u_vals[li*2+lj][r]
                    
                    gi = b_rows[li]
                    gj = b_cols[lj]
                    v[gi * p + gj] = v_vals[li*2+lj][r]
                    
                    gi = c_rows[li]
                    gj = c_cols[lj]
                    w[gi * n + gj] = w_vals[li*2+lj][r]
            
            U_block.append(u)
            V_block.append(v)
            W_block.append(w)

    # 1. Top-Left Block (2x4 * 4x4)
    # A: rows 0,1; cols 0,1 (Block 1) and rows 0,1; cols 2,3 (Block 2)
    # B: rows 0,1; cols 0,1 (Block 11), rows 2,3; cols 0,1 (Block 21)
    #    rows 0,1; cols 2,3 (Block 12), rows 2,3; cols 2,3 (Block 22)
    
    # C: rows 0,1; cols 0,1 (Block 1) and rows 0,1; cols 2,3 (Block 2)
    
    # Term 1: A1 B11
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # Term 2: A2 B21
    add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # Term 3: A1 B12
    add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # Term 4: A2 B22
    add_strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # 2. Top-Right (2x4 * 4x1)
    # Schoolbook for this part
    for i in [0, 1]:
        for j in range(4):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i*m+j] = 1.0
            v[j*p+0] = 1.0
            w[0*n+i] = 1.0
            U_block.append(u); V_block.append(v); W_block.append(w)
            
    # 3. Bottom-Left (1x4 * 4x4)
    for i in [2]:
        for j in range(4):
            for k in range(4):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m+j] = 1.0
                v[j*p+k] = 1.0
                w[k*n+i] = 1.0
                U_block.append(u); V_block.append(v); W_block.append(w)
                
    # 4. Bottom-Right (1x4 * 4x1)
    for i in [2]:
        for j in range(4):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i*m+j] = 1.0
            v[j*p+4] = 1.0
            w[4*n+i] = 1.0
            U_block.append(u); V_block.append(v); W_block.append(w)
            
    U_init = np.array(U_block).T
    V_init = np.array(V_block).T
    W_init = np.array(W_block).T
    
    if verify(U_init, V_init, W_init):
        best_U, best_V, best_W = U_init, V_init, W_init
        best_rank = U_init.shape[1]
        
    # Optimize
    for restart in range(5):
        if time.time() > deadline - 10: break
        U = best_U.copy() + np.random.randn(*best_U.shape)*0.01
        V = best_V.copy() + np.random.randn(*best_V.shape)*0.01
        W = best_W.copy() + np.random.randn(*best_W.shape)*0.01
        
        for step in range(100):
            T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = T_approx - T
            dU = np.einsum('abc,br,cr->ar', diff, V, W)
            dV = np.einsum('abc,ar,cr->br', diff, U, W)
            dW = np.einsum('abc,ar,br->cr', diff, U, V)
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if verify(U, V, W):
                # Try to round
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = U_r.shape[1]
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
