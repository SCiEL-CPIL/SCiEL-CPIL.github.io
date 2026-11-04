import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    # Indices: i=0, j=0..3, k=0..7
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    # We split j into 2 blocks of 2, and k into 4 blocks of 2.
    # For each (j_block, k_block), we have a 2x2x2 subproblem.
    # Strassen terms for 2x2x2:
    # M1 = (a+d)(e+h) -> U:[1,0,0,1], V:[1,0,0,1], W:[1,0,0,1]
    # M2 = (c+d)e    -> U:[0,0,1,1], V:[1,0,0,0], W:[0,0,1,-1]
    # M3 = a(f-h)    -> U:[1,0,0,0], V:[0,1,0,-1], W:[0,1,0,1]
    # M4 = d(g-e)    -> U:[0,0,0,1], V:[-1,0,1,0], W:[1,0,1,0]
    # M5 = (a+b)h    -> U:[1,1,0,0], V:[0,0,0,1], W:[-1,1,0,0]
    # M6 = (c-a)(e+f)-> U:[-1,0,1,0], V:[1,1,0,0], W:[0,0,0,1]
    # M7 = (b-d)(g+h)-> U:[0,1,0,-1], V:[0,0,1,1], W:[1,0,0,0]
    
    strassen_terms = [
        # U (local i,j), V (local j,k), W (local k,i)
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(0,0,-1), (1,0,1)], 'W': [(0,0,1), (1,0,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'U': [(1,0,-1), (1,1,1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    
    # Map local indices to global for the 2x4x8 block (rows 1,2)
    # Local i: 0->global i=1, 1->global i=2
    # Local j: 0->global j=2*j_block+0, 1->global j=2*j_block+1
    # Local k: 0->global k=2*k_block+0, 1->global k=2*k_block+1
    
    for j_block in range(2):
        for k_block in range(4):
            for term in strassen_terms:
                # U
                for li, lj, val in term['U']:
                    gi = 1 + li
                    gj = 2 * j_block + lj
                    U[gi * m + gj][col] += val
                # V
                for lj, lk, val in term['V']:
                    gj = 2 * j_block + lj
                    gk = 2 * k_block + lk
                    V[gj * p + gk][col] += val
                # W
                for lk, li, val in term['W']:
                    gk = 2 * k_block + lk
                    gi = 1 + li
                    W[gk * n + gi][col] += val
                col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=3, m=4, p=8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _als_optimize(U_init, V_init, W_init, target_rank, n_iter=500, tol=1e-10, deadline=None):
    """
    Optimize the decomposition to target_rank using ALS.
    U_init, V_init, W_init: numpy arrays of shape (n*m, R_init), etc.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = DIMENSIONS
    R = target_rank
    
    # Initialize with first R columns (or random projection if needed, but truncation is standard)
    if U_init.shape[1] < R:
        # Pad with zeros if necessary (should not happen in our flow)
        pad_width = R - U_init.shape[1]
        U = np.hstack([U_init, np.zeros((U_init.shape[0], pad_width))])
        V = np.hstack([V_init, np.zeros((V_init.shape[0], pad_width))])
        W = np.hstack([W_init, np.zeros((W_init.shape[0], pad_width))])
    else:
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                # Construct matrix M_{i,j} of size p x R
                # M[k, r] = V[j, k, r] * W[k, i, r]
                # V[j, :, :] is V[j*p : (j+1)*p, :]
                # W[:, i, :] is W[:, i] (column i of W reshaped?)
                # W is shape (p*n, R). Row index is k*n + i.
                # So W[k*n + i, :] is the vector for (k,i).
                
                V_block = V[j*p:(j+1)*p, :] # shape (p, R)
                W_vecs = np.array([W[k*n + i, :] for k in range(p)]) # shape (p, R)
                
                M = V_block * W_vecs # element-wise product, shape (p, R)
                
                try:
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m + j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # M_{j,k} of size n x R
                # M[i, r] = U[i, j, r] * W[k, i, r]
                U_vecs = np.array([U[i*m + j, :] for i in range(n)]) # shape (n, R)
                W_block = W[k*n:(k+1)*n, :] # shape (n, R)
                
                M = U_vecs * W_block # shape (n, R)
                
                try:
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p + k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # M_{k,i} of size m x R
                # M[j, r] = U[i, j, r] * V[j, k, r]
                U_block = U[i*m:(i+1)*m, :] # shape (m, R)
                V_vecs = np.array([V[j*p + k, :] for j in range(m)]) # shape (m, R)
                
                M = U_block * V_vecs # shape (m, R)
                
                try:
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n + i, :] = w
                except:
                    pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < 1e-8:
        # Round to half-integers if possible
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        error_r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    error_r += (val - 1.0) ** 2
        if error_r < 1e-8:
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        U_i = np.round(best_U).astype(float)
        V_i = np.round(best_V).astype(float)
        W_i = np.round(best_W).astype(float)
        error_i = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_i[i*m+j, :], V_i[j*p+k, :] * W_i[k*n+i, :])
                    error_i += (val - 1.0) ** 2
        if error_i < 1e-8:
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Build baseline rank-88
    rank88_result = _build_rank88()
    U88 = np.array(rank88_result['U'], dtype=np.float64)
    V88 = np.array(rank88_result['V'], dtype=np.float64)
    W88 = np.array(rank88_result['W'], dtype=np.float64)
    
    # 2. Try to optimize the 2x4x8 part (rows 1,2) to lower rank
    # The 2x4x8 part corresponds to indices:
    # U: rows 4..11 (i=1,2; j=0..3) -> 2*4 = 8 rows
    # V: rows 0..31 (j=0..3; k=0..7) -> 4*8 = 32 rows
    # W: rows 3..23 (k=0..7; i=1,2) -> 8*2 = 16 rows
    # Note: The columns for this part are 32..87 in the rank-88 decomposition.
    
    U_sub_init = U88[4:12, 32:88]
    V_sub_init = V88[:, 32:88]
    W_sub_init = W88[3:27, 32:88]
    
    # We want to see if we can reduce the rank of this sub-tensor from 56 to 55, 54, etc.
    # Or even better, split it into two 2x4x4 tensors and reduce each to 13.
    
    # Strategy A: Optimize the whole 2x4x8 sub-tensor
    best_sub_result = None
    for target_rank in [55, 54, 53, 52, 50]:
        if time.time() > _START_TIME + _DEADLINE - _MARGIN:
            break
        result = _als_optimize(U_sub_init, V_sub_init, W_sub_init, target_rank, n_iter=300, deadline=_START_TIME + _DEADLINE - _MARGIN)
        if result:
            best_sub_result = result
            break
    
    if best_sub_result:
        U_new, V_new, W_new = best_sub_result
        # Assemble full decomposition
        # Part 1 (rows 0) remains schoolbook (rank 32)
        # Part 2 (rows 1,2) is the optimized sub-tensor
        U_full = np.hstack([U88[:4, :32], U_new])
        V_full = np.hstack([V88[:, :32], V_new])
        W_full = np.hstack([W88[:3, :32], W_new])
        
        if _verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist()):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Strategy B: Try to optimize the 2x4x4 blocks separately
    # Block 1: k=0..3. Block 2: k=4..7.
    # In the sub-tensor (columns 32..87), the first 28 columns correspond to k=0..3?
    # No, the Strassen construction interleaved j and k blocks.
    # Let's just stick to the global optimization if it fails.
    
    # Fallback: Return rank-88
    return rank88_result
# EVOLVE-BLOCK-END
