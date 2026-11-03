import random
import time
import numpy as np
from itertools import product

_START = time.time()
_DEADLINE = 135.0

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p, tol=1e-8):
    """Check if the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _get_strassen_terms():
    """
    Return the local U, V, W vectors for the 7 Strassen terms for 2x2x2 matrix multiplication.
    Local indices 0, 1.
    U is for A (2x2), V is for B (2x2), W is for C (2x2).
    U index: i*2 + j
    V index: j*2 + k
    W index: k*2 + i
    """
    terms = []
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += 1, C11 += 1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # W00=1 (k0,i0), W11=1 (k1,i1). W01, W10 are 0.
    terms.append((u1, v1, w1))
    
    # M2 = (A10 + A11)B00
    # C10 += 1, C11 -= 1
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # W10=1 (k0,i1), W11=-1 (k1,i1).
    terms.append((u2, v2, w2))
    
    # M3 = A00(B01 - B11)
    # C01 += 1, C11 += 1
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # W01=1 (k1,i0), W11=1 (k1,i1).
    terms.append((u3, v3, w3))
    
    # M4 = A11(B10 - B00)
    # C00 += 1, C10 += 1
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0] # W00=1 (k0,i0), W10=1 (k0,i1).
    terms.append((u4, v4, w4))
    
    # M5 = (A00 + A01)B11
    # C00 -= 1, C01 += 1
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # W00=-1 (k0,i0), W01=1 (k1,i0).
    terms.append((u5, v5, w5))
    
    # M6 = (A10 - A00)(B00 + B01)
    # C11 += 1
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # W11=1 (k1,i1).
    terms.append((u6, v6, w6))
    
    # M7 = (A01 - A11)(B10 + B11)
    # C00 += 1
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # W00=1 (k0,i0).
    terms.append((u7, v7, w7))
    
    return terms

def _find_max_disjoint_blocks(n, m, p):
    """
    Find the maximum number of disjoint 2x2x2 blocks in an n x m x p tensor.
    Returns a list of blocks, where each block is (I, J, K) with I, J, K being lists of 2 indices.
    """
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i1 in range(n-1):
        for j1 in range(m-1):
            for k1 in range(p-1):
                I = [i1, i1+1]
                J = [j1, j1+1]
                K = [k1, k1+1]
                blocks.append((I, J, K))
    
    # We want to select the maximum number of blocks such that no two blocks share any (i,j,k) index.
    # This is a maximum independent set problem in a conflict graph, which is NP-hard.
    # However, the graph is small, so we can use a greedy approach or backtracking.
    # Given the small size, we can try all subsets if the number of blocks is small, or use a heuristic.
    # Number of blocks: (3-1)*(5-1)*(7-1) = 2*4*6 = 48.
    # 2^48 is too large. We need a heuristic.
    
    # Greedy approach:
    # Sort blocks by some priority (e.g., random, or by index sum)
    # Select a block if it doesn't conflict with already selected blocks.
    
    # Let's try a simple greedy with random restarts.
    best_blocks = []
    best_count = 0
    
    for _ in range(100):
        if _elapsed() > _DEADLINE - 5:
            break
            
        random.shuffle(blocks)
        selected = []
        used_indices = set()
        
        for I, J, K in blocks:
            conflict = False
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in used_indices:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if not conflict:
                selected.append((I, J, K))
                for i in I:
                    for j in J:
                        for k in K:
                            used_indices.add((i, j, k))
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_blocks = selected
            
    return best_blocks

def construct():
    global _START
    _START = time.time()
    
    n, m, p = 3, 5, 7
    
    # Find the best set of disjoint Strassen blocks
    blocks = _find_max_disjoint_blocks(n, m, p)
    
    # If no blocks found, fall back to schoolbook
    if not blocks:
        rank_school = n*m*p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Initialize U, V, W with zeros
    # We don't know the rank yet, so we use a dictionary or list of lists
    # Let's use a list of columns, where each column is a (U_col, V_col, W_col)
    columns = []
    
    # Track which (i,j,k) tuples are covered
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    strassen_terms = _get_strassen_terms()
    
    # Process Strassen blocks
    for I, J, K in blocks:
        for u_loc, v_loc, w_loc in strassen_terms:
            u_col = [0.0] * (n*m)
            v_col = [0.0] * (m*p)
            w_col = [0.0] * (p*n)
            
            for idx_u, val_u in enumerate(u_loc):
                if val_u != 0:
                    i_loc = idx_u // 2
                    j_loc = idx_u % 2
                    i_g = I[i_loc]
                    j_g = J[j_loc]
                    u_col[i_g*m + j_g] += val_u
            
            for idx_v, val_v in enumerate(v_loc):
                if val_v != 0:
                    j_loc = idx_v // 2
                    k_loc = idx_v % 2
                    j_g = J[j_loc]
                    k_g = K[k_loc]
                    v_col[j_g*p + k_g] += val_v
            
            for idx_w, val_w in enumerate(w_loc):
                if val_w != 0:
                    k_loc = idx_w // 2
                    i_loc = idx_w % 2
                    k_g = K[k_loc]
                    i_g = I[i_loc]
                    w_col[k_g*n + i_g] += val_w
            
            columns.append((u_col, v_col, w_col))
    
    # Process remaining terms (Schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_col = [0.0] * (n*m)
                    v_col = [0.0] * (m*p)
                    w_col = [0.0] * (p*n)
                    u_col[i*m + j] = 1.0
                    v_col[j*p + k] = 1.0
                    w_col[k*n + i] = 1.0
                    columns.append((u_col, v_col, w_col))
    
    rank = len(columns)
    
    # Convert to lists of lists
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for col_idx, (u_col, v_col, w_col) in enumerate(columns):
        for idx, val in enumerate(u_col):
            U[idx][col_idx] = val
        for idx, val in enumerate(v_col):
            V[idx][col_idx] = val
        for idx, val in enumerate(w_col):
            W[idx][col_idx] = val
    
    # Convert to int if possible
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    # Validate
    if _validate(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        rank_school = n*m*p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}
