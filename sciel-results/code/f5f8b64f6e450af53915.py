import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

def _rank33_solution():
    """
    Build a rank-33 decomposition using 3 blocks of a 2x2-by-2x2 Strassen-like 
    decomposition with 11 terms per block.
    """
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 11 terms for a 2x2 * 2x2 multiplication using Strassen-style with 11 products.
    # Standard Strassen has 7 products, but the 11-term scheme here comes from 
    # a specific decomposition. Let's use the 11 terms from the current program.
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}

def _try_rank32_search():
    """
    Attempt to find a rank-32 solution by modifying the rank-33 solution.
    Strategy: Try to express one column of the rank-33 decomposition as a linear
    combination of the others, effectively reducing the rank by 1.
    """
    n, m, p = DIMENSIONS
    base = _rank33_solution()
    U33 = np.array(base['U'], dtype=float)
    V33 = np.array(base['V'], dtype=float)
    W33 = np.array(base['W'], dtype=float)
    
    # We want to see if any column is a linear combination of the others.
    # For each column c, check if the 36x1 vector formed by the tensor contributions
    # of column c is in the span of the other 35 columns.
    # The tensor contribution of column r is a 36-dimensional vector T_r where
    # T_r[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def tensor_col(r):
        vec = np.zeros(36)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    vec[idx] = U33[i*m + j, r] * V33[j*p + k, r] * W33[k*n + i, r]
        return vec
    
    # Try removing each column and see if the remaining 32 columns can still span
    # the full 36-dimensional tensor space.
    for remove_col in range(33):
        # Build the matrix of the other 32 columns
        cols_to_keep = [c for c in range(33) if c != remove_col]
        if len(cols_to_keep) != 32:
            continue
        
        # Check if the tensor of the removed column is in the span of the other 32
        target = tensor_col(remove_col)
        
        # Build a 36x32 matrix of the other columns
        M = np.zeros((36, 32))
        for col_idx, c in enumerate(cols_to_keep):
            M[:, col_idx] = tensor_col(c)
        
        # Check if target is in the column space of M
        # Use SVD to check
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # Project target onto the column space
            # If the projection is close to target, then it's in the span
            proj = U_svd @ (U_svd.T @ target)
            residual = target - proj
            if np.linalg.norm(residual) < 1e-8:
                # Column remove_col is redundant
                # Build the rank-32 solution
                U32 = np.array(base['U'], dtype=float)[:, cols_to_keep]
                V32 = np.array(base['V'], dtype=float)[:, cols_to_keep]
                W32 = np.array(base['W'], dtype=float)[:, cols_to_keep]
                
                # Convert back to lists
                result = {
                    "U": U32.tolist(),
                    "V": V32.tolist(),
                    "W": W32.tolist()
                }
                
                # Verify
                if _verify(result["U"], result["V"], result["W"]):
                    return result
        except np.linalg.LinAlgError:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Start with the rank-33 solution
    best_result = _rank33_solution()
    
    # Try to find a rank-32 solution
    if time.time() < deadline:
        rank32_result = _try_rank32_search()
        if rank32_result is not None:
            best_result = rank32_result
    
    # Verify the best result before returning
    if not _verify(best_result["U"], best_result["V"], best_result["W"]):
        # Fallback to the rank-33 solution if verification fails
        best_result = _rank33_solution()
    
    return best_result
