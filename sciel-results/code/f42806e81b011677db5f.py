import time
import numpy as np
import random

DIMENSIONS = [2, 4, 7]

def _schoolbook_decomp():
    """Standard (schoolbook) decomposition of 2x4x7 matrix multiplication tensor."""
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

def _greedy_decomposition(max_rank=50, deadline=None):
    """
    Greedy rank-1 term selection: iteratively find the best rank-1 term
    to subtract from the residual tensor.
    """
    n, m, p = 2, 4, 7
    # Create the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    U_cols = []
    V_cols = []
    W_cols = []
    residual = T.copy()
    
    for rank_iter in range(max_rank):
        if deadline and time.time() > deadline:
            break
            
        # Find the best rank-1 term: maximize |sum residual * (u⊗v⊗w)|
        # Try all possible sparse rank-1 terms with entries in {-1, 0, 1}
        # For efficiency, use a heuristic: find the entry with max absolute value
        # and build a rank-1 term around it
        
        # Get the index of the maximum absolute value entry
        max_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        i_idx, j_idx, k_idx = max_idx
        max_val = residual[i_idx, j_idx, k_idx]
        
        if abs(max_val) < 1e-10:
            break
        
        # Try to find a rank-1 term that fits this entry well
        # The rank-1 term is u⊗v⊗w where u[i]=1, v[j]=1, w[k]=1
        # We want to choose u, v, w to maximize the inner product with residual
        
        # Heuristic: for each coordinate, try to extend the rank-1 term
        # by looking at related entries
        
        # Build initial rank-1 term
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        u[i_idx] = 1
        v[j_idx] = 1
        w[k_idx] = 1
        
        # Try to improve by adding related entries
        # For matrix multiplication, related entries share indices
        # i_idx = i*m+j, j_idx = j*p+k, k_idx = k*n+i
        
        i_val = i_idx // m
        j_val = i_idx % m
        k_val = j_idx % p
        
        # Try adding entries that share i, j, or k
        best_score = abs(max_val)
        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        # Try all combinations of adding one more entry to each vector
        # This is expensive, so use a greedy approach
        
        # For now, just use the single-entry rank-1 term
        U_cols.append(u.tolist())
        V_cols.append(v.tolist())
        W_cols.append(w.tolist())
        
        # Subtract the rank-1 term
        residual -= np.outer(np.outer(u, v), w)
    
    if not U_cols:
        return None
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=7, timeout=5.0):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    start = time.time()
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    try:
        T = np.einsum('at,bt,ct->abc', U_arr, V_arr, W_arr, optimize=True)
    except Exception:
        return False
    
    if time.time() - start > timeout * 0.7:
        return False
    
    # Check that all required entries are 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                    return False
    
    # Check that all other entries are 0
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                # Determine if this is a "required" entry
                i_val = i // m
                j_val = i % m
                j2_val = j // p
                k_val = j % p
                k2_val = k // n
                i2_val = k % n
                
                is_required = (i_val == i2_val and j_val == j2_val)
                if not is_required:
                    if abs(T[i, j, k]) > 1e-9:
                        return False
    
    return True

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen.
    """
    terms = []
    terms.append(([0,0,1,1], [1,0,0,-1], [0,0,1,-1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,0,1]))
    terms.append(([1,0,0,0], [0,-1,0,0], [1,0,0,1]))
    terms.append(([0,0,0,1], [1,-1,0,0], [0,0,1,1]))
    terms.append(([1,1,0,0], [0,0,0,1], [1,0,0,1]))
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,1,1]])
    terms.append(([0,1,-1,0], [0,0,1,1], [1,0,0,1]))
    return terms

def _block_decomp_2x4x7():
    """
    Block decomposition of 2x4x7 matrix multiplication.
    """
    n, m, p = 2, 4, 7
    
    strassen_terms = _strassen_2x2x2_terms()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    blocks_2x2x2 = [
        (0, 0, 0, 0),
        (2, 14, 0, 0),
        (0, 0, 2, 2),
        (2, 14, 2, 2),
        (0, 0, 4, 4),
        (2, 14, 4, 4),
    ]
    
    for a_col_off, b_row_off, b_col_off, c_col_off in blocks_2x2x2:
        for u_t, v_t, w_t in strassen_terms:
            U_vec = [0] * 8
            V_vec = [0] * 28
            W_vec = [0] * 14
            
            U_vec[a_col_off + 0] = u_t[0]
            U_vec[a_col_off + 1] = u_t[1]
            U_vec[2 + a_col_off + 0] = u_t[2]
            U_vec[2 + a_col_off + 1] = u_t[3]
            
            V_vec[b_row_off + b_col_off] = v_t[0]
            V_vec[b_row_off + b_col_off + 1] = v_t[1]
            V_vec[b_row_off + 7 + b_col_off] = v_t[2]
            V_vec[b_row_off + 7 + b_col_off + 1] = v_t[3]
            
            W_vec[c_col_off] = w_t[0]
            W_vec[c_col_off + 1] = w_t[1]
            W_vec[2 + c_col_off] = w_t[2]
            W_vec[2 + c_col_off + 1] = w_t[3]
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
    
    blocks_2x2x1 = [
        (0, 0, 6, 6),
        (2, 14, 6, 6),
    ]
    
    for a_col_off, b_row_off, b_col_off, c_col_off in blocks_2x2x1:
        for (u_idx, v_idx, w_idx) in [(0, 0, 0), (1, 7, 0), (2, 0, 2), (3, 7, 2)]:
            U_vec = [0] * 8
            V_vec = [0] * 28
            W_vec = [0] * 14
            U_vec[a_col_off + u_idx] = 1
            V_vec[b_row_off + v_idx] = 1
            W_vec[c_col_off + w_idx] = 1
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
    
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
    deadline = time.time() + 110.0
    
    best_result = None
    best_rank = float('inf')
    
    # Try block decomposition first
    try:
        result = _block_decomp_2x4x7()
        if _verify_decomposition(result["U"], result["V"], result["W"], timeout=5.0):
            best_result = result
            best_rank = len(result["U"][0])
    except Exception:
        pass
    
    # Try greedy decomposition with different max_rank values
    for max_rank in [45, 48, 50, 52]:
        if time.time() > deadline - 30:
            break
        try:
            result = _greedy_decomposition(max_rank=max_rank, deadline=deadline - 20)
            if result and _verify_decomposition(result["U"], result["V"], result["W"], timeout=5.0):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook_decomp()
