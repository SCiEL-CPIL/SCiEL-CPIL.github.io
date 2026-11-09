import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen.
    """
    terms = []
    # M1 = (c+d)*e
    terms.append(([0,0,1,1], [1,0,0,0], [1,0,0,1]))
    # M2 = (c+d)*(f-h)
    terms.append(([0,0,1,1], [0,1,0,-1], [0,0,1,-1]))
    # M3 = a*(e-f)
    terms.append(([1,0,0,0], [1,-1,0,0], [-1,1,0,0]))
    # M4 = d*(g-h)
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    # M5 = (a+b)*h
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6 = (c-a)*(e+f)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)*(g+h)
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,1]))
    return terms

def _schoolbook_decomp():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_decomp_2x4x7():
    n, m, p = 2, 4, 7
    strassen_terms = _strassen_2x2x2_terms()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Blocks of 2x2x2
    # A parts: cols 0-1 (offset 0), cols 2-3 (offset 2)
    # B parts: rows 0-1 (offset 0), rows 2-3 (offset 14)
    # C parts: cols 0-1 (offset 0), 2-3 (offset 2), 4-5 (offset 4)
    
    blocks_2x2x2 = [
        (0, 0, 0), (2, 14, 0),
        (0, 0, 2), (2, 14, 2),
        (0, 0, 4), (2, 14, 4)
    ]
    
    for a_off, b_off, c_off in blocks_2x2x2:
        for u_t, v_t, w_t in strassen_terms:
            U_vec = [0] * 8
            V_vec = [0] * 28
            W_vec = [0] * 14
            
            # A mapping
            U_vec[a_off] = u_t[0]
            U_vec[a_off + 1] = u_t[1]
            U_vec[2 + a_off] = u_t[2]
            U_vec[2 + a_off + 1] = u_t[3]
            
            # B mapping
            V_vec[b_off] = v_t[0]
            V_vec[b_off + 1] = v_t[1]
            V_vec[b_off + 7] = v_t[2]
            V_vec[b_off + 8] = v_t[3]
            
            # C mapping
            W_vec[c_off] = w_t[0]
            W_vec[c_off + 1] = w_t[1]
            W_vec[2 + c_off] = w_t[2]
            W_vec[2 + c_off + 1] = w_t[3]
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
            
    # Blocks of 2x2x1
    # A parts: cols 0-1 (offset 0), cols 2-3 (offset 2)
    # B parts: rows 0-1 (offset 0), rows 2-3 (offset 14), col 6
    # C parts: col 6 (offset 6)
    
    blocks_2x2x1 = [
        (0, 0, 6),
        (2, 14, 6)
    ]
    
    for a_off, b_off, c_off in blocks_2x2x1:
        # 4 terms for 2x2x1
        # 1. A[0,0] * B[0,0] -> C[0]
        U_vec = [0]*8; V_vec = [0]*28; W_vec = [0]*14
        U_vec[a_off] = 1
        V_vec[b_off] = 1
        W_vec[c_off] = 1
        U_cols.append(U_vec); V_cols.append(V_vec); W_cols.append(W_vec)
        
        # 2. A[0,1] * B[1,0] -> C[0]
        U_vec = [0]*8; V_vec = [0]*28; W_vec = [0]*14
        U_vec[a_off+1] = 1
        V_vec[b_off+7] = 1
        W_vec[c_off] = 1
        U_cols.append(U_vec); V_cols.append(V_vec); W_cols.append(W_vec)
        
        # 3. A[1,0] * B[0,0] -> C[1]
        U_vec = [0]*8; V_vec = [0]*28; W_vec = [0]*14
        U_vec[2+a_off] = 1
        V_vec[b_off] = 1
        W_vec[2+c_off] = 1
        U_cols.append(U_vec); V_cols.append(V_vec); W_cols.append(W_vec)
        
        # 4. A[1,1] * B[1,0] -> C[1]
        U_vec = [0]*8; V_vec = [0]*28; W_vec = [0]*14
        U_vec[2+a_off+1] = 1
        V_vec[b_off+7] = 1
        W_vec[2+c_off] = 1
        U_cols.append(U_vec); V_cols.append(V_vec); W_cols.append(W_vec)
        
    rank = len(U_cols)
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(14)]
    
    for col in range(rank):
        for row in range(8):
            U[row][col] = U_cols[col][row]
        for row in range(28):
            V[row][col] = V_cols[col][row]
        for row in range(14):
            W[row][col] = W_cols[col][row]
            
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 115.0
    try:
        result = _block_decomp_2x4x7()
        
        # Verify
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        n, m, p = 2, 4, 7
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    err = abs(expected - actual)
                    if err > max_err: max_err = err
        
        if max_err < 1e-10:
            return result
        else:
            return _schoolbook_decomp()
    except Exception:
        return _schoolbook_decomp()
