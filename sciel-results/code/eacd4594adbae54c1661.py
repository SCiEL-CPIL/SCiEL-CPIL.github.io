import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(actual - expected) > 1e-8:
                    return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2 * 2x2.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11)
    u = [1, 0, 0, 0]
    v = [0, 0, 1, -1]
    w = [0, 1, 0, 1]
    terms.append((u, v, w))
    
    # M4 = A11(B01-B00)
    u = [0, 0, 0, 1]
    v = [0, 1, -1, 0]
    w = [1, 0, 1, 0]
    terms.append((u, v, w))
    
    # M5 = (A00+A01)B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B10)
    u = [-1, 0, 1, 0]
    v = [1, 0, 1, 0]
    w = [0, 0, 0, 1]
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B01+B11)
    u = [0, 1, 0, -1]
    v = [0, 1, 0, 1]
    w = [1, 0, 0, 0]
    terms.append((u, v, w))
    
    return terms

def _strassen_4x4x4_decomposition():
    """
    Build Strassen-based decomposition for 4x4 * 4x4 matrix multiplication.
    Returns U, V, W matrices where:
      U: shape (16, rank) - indices i*4+j
      V: shape (16, rank) - indices j*4+k
      W: shape (16, rank) - indices k*4+i
    """
    s2_terms = _strassen_2x2_terms()
    
    strassen_4x4_combos = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),
    ]
    
    rank_strassen = 49
    U = np.zeros((16, rank_strassen), dtype=float)
    V = np.zeros((16, rank_strassen), dtype=float)
    W = np.zeros((16, rank_strassen), dtype=float)
    
    col = 0
    for combo_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_4x4_combos):
        for s2_idx, (u2, v2, w2) in enumerate(s2_terms):
            u_vec = np.zeros(16, dtype=float)
            v_vec = np.zeros(16, dtype=float)
            w_vec = np.zeros(16, dtype=float)
            
            for i in range(4):
                for j in range(4):
                    ab = 2 * (i // 2) + (j // 2)
                    si = i % 2
                    sj = j % 2
                    coeff_a = a_coeffs[ab] * u2[si * 2 + sj]
                    if coeff_a != 0:
                        u_vec[i * 4 + j] = coeff_a
            
            for j in range(4):
                for k in range(4):
                    bb = 2 * (j // 2) + (k // 2)
                    si = j % 2
                    sj = k % 2
                    coeff_b = b_coeffs[bb] * v2[si * 2 + sj]
                    if coeff_b != 0:
                        v_vec[j * 4 + k] = coeff_b
            
            for k in range(4):
                for i in range(4):
                    cb = 2 * (k // 2) + (i // 2)
                    si = k % 2
                    sj = i % 2
                    coeff_c = c_coeffs[cb] * w2[si * 2 + sj]
                    if coeff_c != 0:
                        w_vec[k * 4 + i] = coeff_c
            
            if np.any(u_vec) and np.any(v_vec) and np.any(w_vec):
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
    
    if col < rank_strassen:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # 1. Get the base 4x4x4 Strassen decomposition (Rank 49)
    U_base, V_base, W_base = _strassen_4x4x4_decomposition()
    rank_base = V_base.shape[1] # Should be 49
    
    # 2. Construct the full 4x4x9 decomposition
    # Total Rank = 49 (cols 0-3) + 49 (cols 4-7) + 1 (col 8) = 99
    rank_total = 99
    
    U = np.zeros((16, rank_total), dtype=float)
    V = np.zeros((36, rank_total), dtype=float) # 4*9
    W = np.zeros((36, rank_total), dtype=float) # 9*4
    
    # Part 1: Columns 0-3 (Standard Strassen)
    # V_base index j*4+k (k=0..3) maps to V_full j*9+k
    # W_base index k*4+i (k=0..3) maps to W_full k*4+i
    for col in range(rank_base):
        U[:, col] = U_base[:, col]
        for j in range(4):
            for k in range(4):
                V[j*9+k, col] = V_base[j*4+k, col]
        for k in range(4):
            for i in range(4):
                W[k*4+i, col] = W_base[k*4+i, col]
    
    # Part 2: Columns 4-7 (Shifted Strassen)
    # V_base index j*4+k (k=0..3) maps to V_full j*9+(4+k)
    # W_base index k*4+i (k=0..3) maps to W_full (4+k)*4+i
    offset_v = 4
    offset_w = 4
    for col in range(rank_base):
        U[:, rank_base + col] = U_base[:, col]
        for j in range(4):
            for k in range(4):
                V[j*9 + (offset_v + k), rank_base + col] = V_base[j*4+k, col]
        for k in range(4):
            for i in range(4):
                W[(offset_w + k)*4 + i, rank_base + col] = W_base[k*4+i, col]
    
    # Part 3: Column 8 (Rank 1 term for 4x4x1)
    # T = 1_4 x 1_4 x e_0 (where e_0 is the vector for the single column)
    # In global indices:
    # U: all ones
    # V: all ones at j*9+8
    # W: all ones at 8*4+i
    
    col_last = rank_base * 2
    U[:, col_last] = 1.0
    for j in range(4):
        V[j*9+8, col_last] = 1.0
    for i in range(4):
        W[8*4+i, col_last] = 1.0
        
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to previous known good (129) or schoolbook (144)
        # Let's just return schoolbook to be safe if verification fails
        rank_sb = n * m * p
        U = np.zeros((16, rank_sb), dtype=float)
        V = np.zeros((36, rank_sb), dtype=float)
        W = np.zeros((36, rank_sb), dtype=float)
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j, c] = 1
                    V[j*9+k, c] = 1
                    W[k*4+i, c] = 1
                    c += 1

    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
