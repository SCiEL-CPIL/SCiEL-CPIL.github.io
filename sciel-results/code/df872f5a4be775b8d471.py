import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_decomposition():
    """
    Returns the coefficients for the Strassen 2x2x2 decomposition.
    Format: list of (u_coeffs, v_coeffs, w_coeffs)
    u_coeffs: dict mapping (i, j) to coeff for A[i, j]
    v_coeffs: dict mapping (j, k) to coeff for B[j, k]
    w_coeffs: dict mapping (k, i) to coeff for C[k, i] -- wait, W is indexed k*n+i?
    The task says W[k*n+i, r]. So W index is k*n + i.
    In 2x2x2, n=2, m=2, p=2.
    A is 2x2. B is 2x2. C is 2x2.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor is T[i, j, k] = delta_{i, j, k} ? No.
    T[i, j, k] = 1 if the term is A[i, j] B[j, k] C[k, i]?
    Let's check the trace convention: W[k*n+i, r].
    The output is C[i, k] (row i, col k).
    The term in the sum is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    Here n=2, m=2, p=2.
    U index: i*2 + j.
    V index: j*2 + k.
    W index: k*2 + i.
    So the term contributes to C[i, k].
    
    Strassen formulas:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B21 - B22)
    M4 = A22 * (B12 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to express each M as a product of linear forms.
    M1: u = { (0,0):1, (1,1):1 }, v = { (0,0):1, (1,1):1 }
    M2: u = { (1,0):1, (1,1):1 }, v = { (0,0):1 }
    ...
    
    Then we sum the contributions to C[i, k].
    """
    terms = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    # M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    # M1 contributes to C11 (+1), C22 (+1)
    w1 = {(0,0): 1, (1,1): 1} # C[0,0] and C[1,1]
    # Wait, C11 is C[0,0]. C22 is C[1,1].
    # So W index for C[0,0] is 0*2+0=0. For C[1,1] is 1*2+1=3.
    # So w1 = {0: 1, 3: 1}
    add_term(u1, v1, {0: 1, 3: 1})
    
    # M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    # M2 contributes to C21 (+1). C[1,0] index 0*2+1=1.
    add_term(u2, v2, {1: 1})
    
    # M3
    u3 = {(0,0): 1}
    v3 = {(1,0): 1, (1,1): -1}
    # M3 contributes to C12 (+1). C[0,1] index 1*2+0=2.
    add_term(u3, v3, {2: 1})
    
    # M4
    u4 = {(1,1): 1}
    v4 = {(0,1): 1, (0,0): -1}
    # M4 contributes to C11 (+1), C21 (+1).
    # C11 index 0. C21 index 1.
    add_term(u4, v4, {0: 1, 1: 1})
    
    # M5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    # M5 contributes to C11 (-1), C12 (+1).
    # C11 index 0. C12 index 2.
    add_term(u5, v5, {0: -1, 2: 1})
    
    # M6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    # M6 contributes to C22 (+1).
    # C22 index 3.
    add_term(u6, v6, {3: 1})
    
    # M7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    # M7 contributes to C11 (+1).
    # C11 index 0.
    add_term(u7, v7, {0: 1})
    
    return terms

def _build_block_3x2x5(offset_i, offset_j, offset_k):
    """
    Builds the decomposition for a 3x2x5 block.
    offset_i: row offset in A (0..2)
    offset_j: col offset in A / row offset in B (0..1)
    offset_k: col offset in B (0..4)
    
    Returns list of terms (u_dict, v_dict, w_dict)
    u_dict keys: (i, j) relative to block
    v_dict keys: (j, k) relative to block
    w_dict keys: (k, i) relative to block? No, W index is k*n+i.
    In the global matrix, W index is (global_k * 3 + global_i).
    """
    terms = []
    
    # Sub-block 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2 Strassen)
    strassen_terms = _strassen_2x2x2_decomposition()
    for u, v, w in strassen_terms:
        # Map to global indices
        gu = {}
        for (i, j), coeff in u.items():
            gu[(offset_i + i, offset_j + j)] = coeff
        gv = {}
        for (j, k), coeff in v.items():
            gv[(offset_j + j, offset_k + k)] = coeff
        gw = {}
        for w_idx, coeff in w.items():
            # w_idx is an integer 0..3 representing local k*2 + i
            local_k = w_idx // 2
            local_i = w_idx % 2
            global_k = offset_k + local_k
            global_i = offset_i + local_i
            # W index is global_k * n + global_i. Here n=3.
            gw[global_k * 3 + global_i] = coeff
        terms.append((gu, gv, gw))
        
    # Sub-block 2: A[2:3, 0:2] * B[0:2, 0:2] (1x2x2 Schoolbook)
    # Rank 4.
    for i in range(1): # local i 0..0 (global 2)
        for j in range(2): # local j 0..1 (global 0,1)
            for k in range(2): # local k 0..1 (global 0,1)
                u = {(offset_i + 2 + i, offset_j + j): 1}
                v = {(offset_j + j, offset_k + k): 1}
                w = {(offset_k + k) * 3 + (offset_i + 2 + i): 1}
                terms.append((u, v, w))
                
    # Sub-block 3: A[0:2, 0:2] * B[0:2, 2:5] (2x2x3 Schoolbook)
    # Rank 12.
    for i in range(2): # local i 0..1
        for j in range(2): # local j 0..1
            for k in range(3): # local k 0..2 (global 2,3,4)
                u = {(offset_i + i, offset_j + j): 1}
                v = {(offset_j + j, offset_k + 2 + k): 1}
                w = {(offset_k + 2 + k) * 3 + (offset_i + i): 1}
                terms.append((u, v, w))
                
    # Sub-block 4: A[2:3, 0:2] * B[0:2, 2:5] (1x2x3 Schoolbook)
    # Rank 6.
    for i in range(1): # local i 0..0
        for j in range(2): # local j 0..1
            for k in range(3): # local k 0..2
                u = {(offset_i + 2 + i, offset_j + j): 1}
                v = {(offset_j + j, offset_k + 2 + k): 1}
                w = {(offset_k + 2 + k) * 3 + (offset_i + 2 + i): 1}
                terms.append((u, v, w))
                
    return terms

def _build_rank_58_decomposition():
    n, m, p = 3, 4, 5
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A cols 0,1. B rows 0,1.
    terms1 = _build_block_3x2x5(0, 0, 0)
    for u, v, w in terms1:
        for (i, j), coeff in u.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v.items():
            V[j * p + k][col] += coeff
        for w_idx, coeff in w.items():
            W[w_idx][col] += coeff
        col += 1
        
    # Block 2: A cols 2,3. B rows 2,3.
    terms2 = _build_block_3x2x5(0, 2, 0) # Wait, offset_i is 0, offset_j is 2, offset_k is 0?
    # A is 3x4. Cols 2,3 correspond to j=2,3.
    # B is 4x5. Rows 2,3 correspond to j=2,3.
    # So offset_j = 2.
    # The block is A[0:3, 2:4] * B[2:4, 0:5].
    # In _build_block_3x2x5, offset_i=0, offset_j=2, offset_k=0.
    terms2 = _build_block_3x2x5(0, 2, 0)
    for u, v, w in terms2:
        for (i, j), coeff in u.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v.items():
            V[j * p + k][col] += coeff
        for w_idx, coeff in w.items():
            W[w_idx][col] += coeff
        col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition_fast(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_res = U_np.reshape(n, m, rank)
    V_res = V_np.reshape(m, p, rank)
    W_res = W_np.reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U_res, V_res, W_res)
    
    return np.allclose(S, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try Rank 58
    result_58 = _build_rank_58_decomposition()
    if _verify_decomposition_fast(result_58["U"], result_58["V"], result_58["W"], n, m, p):
        return result_58
        
    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
