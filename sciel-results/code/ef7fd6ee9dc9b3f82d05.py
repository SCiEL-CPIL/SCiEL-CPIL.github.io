import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _get_rank23_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication with rank 23.
    U: 9x23, row i = 3*r + c (A_rc)
    V: 9x23, row j = 3*r + c (B_rc)
    W: 9x23, row k = 3*c + r (C_rc) -> Column-major convention: W[col*3 + row]
    """
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Terms from inspiration (Bini-Romani style)
    # Format: (row, col, coeff)
    # U/V indices are row-major (row*3+col)
    # W indices are col-major (col*3+row) in the assignment below
    
    terms = [
        {'U': [(0,0,1), (2,2,1)], 'V': [(0,0,1), (2,2,1)], 'W': [(0,0,1), (2,2,1)]},
        {'U': [(2,0,1), (2,2,1)], 'V': [(0,0,1)], 'W': [(2,0,1), (2,2,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (2,1,-1)], 'W': [(0,1,1), (2,1,1)]},
        {'U': [(2,2,1)], 'V': [(2,0,1), (0,0,-1)], 'W': [(0,2,1), (2,0,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(2,2,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'U': [(2,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(2,1,1)]},
        {'U': [(0,1,1), (2,2,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1), (2,2,1)], 'V': [(0,2,1)], 'W': [(0,2,1), (2,2,1)]},
        {'U': [(2,2,1)], 'V': [(0,2,1), (2,2,-1)], 'W': [(0,2,-1), (2,2,1)]},
        {'U': [(0,0,1), (0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1), (0,2,1)]},
        {'U': [(2,0,1)], 'V': [(0,2,1), (2,0,-1)], 'W': [(2,0,1), (2,2,1)]},
        {'U': [(0,2,1), (2,0,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,2,1)]},
        {'U': [(1,1,1)], 'V': [(1,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,1)], 'V': [(1,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(1,1,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'U': [(1,0,1)], 'V': [(1,1,1), (0,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(0,1,1), (1,0,1)], 'V': [(1,1,1)], 'W': [(0,1,-1), (1,0,1)]},
        {'U': [(1,1,1), (1,0,-1)], 'V': [(0,1,1), (1,0,1)], 'W': [(1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(1,2,1), (1,1,-1)]},
        {'U': [(1,1,1)], 'V': [(1,2,1), (0,2,-1)], 'W': [(0,2,1), (1,2,1)]},
        {'U': [(0,2,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(0,2,-1), (1,2,1)]},
        {'U': [(1,2,1)], 'V': [(1,1,1), (0,2,-1)], 'W': [(1,2,1), (1,1,1)]},
        {'U': [(0,2,1), (1,2,-1)], 'V': [(1,1,1), (1,2,1)], 'W': [(0,2,1)]},
    ]
    
    for r, term in enumerate(terms):
        for lr, lc, coeff in term['U']:
            U[lr * 3 + lc, r] += coeff
        for lr, lc, coeff in term['V']:
            V[lr * 3 + lc, r] += coeff
        for lr, lc, coeff in term['W']:
            # W index: col * 3 + row
            W[lc * 3 + lr, r] += coeff
            
    return U, V, W

def _build_6x6_rank184():
    """
    Builds 6x6 multiplication using 2x2 grid of 3x3 blocks with rank-23 decomposition.
    """
    n, m, p = 6, 6, 6
    r3 = 23
    total_rank = 8 * r3
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    U3, V3, W3 = _get_rank23_3x3()
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                block_id = (bi * 2 + bj) * 2 + bk
                col_offset = block_id * r3
                
                for r in range(r3):
                    # U: A[bi,bk]
                    for i in range(3):
                        for j in range(3):
                            local_u_row = i * 3 + j
                            global_u_row = (3 * bi + i) * m + (3 * bk + j)
                            U[global_u_row, col_offset + r] += U3[local_u_row, r]
                            
                    # V: B[bk,bj]
                    for j in range(3):
                        for k in range(3):
                            local_v_row = j * 3 + k
                            global_v_row = (3 * bk + j) * p + (3 * bj + k)
                            V[global_v_row, col_offset + r] += V3[local_v_row, r]
                            
                    # W: C[bi,bj]
                    # W index convention: k_global = p * col_C + row_C
                    for i in range(3):
                        for k in range(3):
                            local_w_row = k * 3 + i
                            global_w_row = p * (3 * bj + k) + (3 * bi + i)
                            W[global_w_row, col_offset + r] += W3[local_w_row, r]
                            
    return U, V, W

def _build_6x6_rank189():
    """
    Builds 6x6 multiplication using 3x3 grid of 2x2 blocks with Strassen (rank 7).
    """
    n, m, p = 6, 6, 6
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        {'U': [(0,0,1),(1,1,1)], 'V': [(0,0,1),(1,1,1)], 'W': [(0,0,1),(1,1,1)]},
        {'U': [(1,0,1),(1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1),(1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1),(1,1,-1)], 'W': [(0,1,1),(1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1),(0,0,-1)], 'W': [(0,0,1),(1,0,1)]},
        {'U': [(0,0,1),(0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1),(0,1,1)]},
        {'U': [(1,0,1),(0,0,-1)], 'V': [(0,0,1),(0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1),(1,1,-1)], 'V': [(1,0,1),(1,1,1)], 'W': [(0,0,1)]},
    ]
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 3 + bj) * 3 + bk
                col_offset = base_col * 7
                
                for r in range(7):
                    term = strassen_terms[r]
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * bi + lr) * m + (2 * bk + lc)
                        U[u_row, col_offset + r] += coeff
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * bk + lr) * p + (2 * bj + lc)
                        V[v_row, col_offset + r] += coeff
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * bj + lc) * n + (2 * bi + lr)
                        W[w_row, col_offset + r] += coeff
                        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify the decomposition using vectorized operations.
    """
    # Construct expected tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        rA, cA = divmod(i, m)
        for j in range(m*p):
            rB, cB = divmod(j, p)
            if cA != rB:
                continue
            for k in range(p*n):
                cC, rC = divmod(k, n)
                if rC == rA and cC == cB:
                    T_exp[i, j, k] = 1.0
                    
    # Compute tensor from decomposition
    T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Check
    if not np.allclose(T_comp, T_exp, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try rank 184
    try:
        res184 = _build_6x6_rank184()
        if res184 is not None:
            U, V, W = res184
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
        
    if time.time() > deadline:
        U, V, W = _build_6x6_rank189()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to rank 189
    U, V, W = _build_6x6_rank189()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": [], "V": [], "W": []}
