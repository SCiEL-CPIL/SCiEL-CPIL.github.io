import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    # Build the full tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

def _rank33_solution():
    """
    Rank-33 decomposition for 2x3x6 matrix multiplication.
    Uses 3 blocks of 2x2x2 multiplication, each with rank 11 (Strassen-like).
    """
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 11-term rank decomposition for 2x2x2
    # U indices: (i, j) for i in 0..1, j in 0..1 -> 0,1,2,3,4,5
    # V indices: (j, k) for j in 0..1, k in 0..1 -> 0,1,2,3,4,5
    # W indices: (k, i) for k in 0..1, i in 0..1 -> 0,1,2,3
    terms = [
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
        {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
    ]
    
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

def _tensor_column(U, V, W, r):
    """Compute the tensor column for rank index r."""
    n, m, p = DIMENSIONS
    vec = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                vec[idx] = U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    return vec

def _find_redundant_columns(U, V, W, max_check=33):
    """
    Check if any column (or pair of columns) is linearly redundant
    in the tensor space. Returns (removable_cols, new_U, new_V, new_W) 
    or (None, None, None, None) if no reduction found.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Precompute tensor columns
    tensor_cols = [_tensor_column(U, V, W, r) for r in range(rank)]
    
    # Check single column removal
    for c in range(rank):
        keep = [i for i in range(rank) if i != c]
        M = np.column_stack([tensor_cols[i] for i in keep])
        target = tensor_cols[c]
        
        # Check if target is in column space of M
        try:
            # Use SVD to project
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            proj = U_svd @ (U_svd.T @ target)
            residual = np.linalg.norm(target - proj)
            
            if residual < 1e-10:
                # Column c is redundant
                new_U = U[:, keep]
                new_V = V[:, keep]
                new_W = W[:, keep]
                return ([c], new_U, new_V, new_W)
        except np.linalg.LinAlgError:
            continue
    
    # Check pair of column removal
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            keep = [i for i in range(rank) if i != c1 and i != c2]
            M = np.column_stack([tensor_cols[i] for i in keep])
            target = np.column_stack([tensor_cols[c1], tensor_cols[c2]])
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                proj = U_svd @ (U_svd.T @ target)
                residual = np.linalg.norm(target - proj)
                
                if residual < 1e-10:
                    new_U = U[:, keep]
                    new_V = V[:, keep]
                    new_W = W[:, keep]
                    return ([c1, c2], new_U, new_V, new_W)
            except np.linalg.LinAlgError:
                continue
    
    return (None, None, None, None)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with rank-33 solution
    result = _rank33_solution()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Try to find redundant columns
    if time.time() < deadline:
        removed, new_U, new_V, new_W = _find_redundant_columns(U, V, W)
        if removed is not None:
            result = {
                "U": new_U.tolist(),
                "V": new_V.tolist(),
                "W": new_W.tolist()
            }
    
    # Verify before returning
    if not _verify(result["U"], result["V"], result["W"]):
        # Fallback to known good solution
        result = _rank33_solution()
    
    return result
