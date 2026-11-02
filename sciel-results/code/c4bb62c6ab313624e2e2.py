import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank for 3x3x4 is 25. Total rank 50.
    # If we can't find 25, try higher.
    
    # Define the 3x3x4 tensor
    # Indices: i in 0..2, j in 0..2, k in 0..3
    # T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k]
    # In the flattened format:
    # U index: i*m + j
    # V index: j*p_block + k  (p_block = 4)
    # W index: k*n + i
    
    p_block = 4
    n_block = 3
    m_block = 3
    
    # Create the tensor T_block of shape (n*m, m*p_block, p_block*n)
    # We will represent it as a list of non-zero coordinates or a dense matrix for ALS.
    # For ALS, we need to compute U @ V^T etc.
    
    # Let's use a dense representation of the tensor as a 3D array, 
    # but since it's small, we can just use the sparsity.
    
    # The tensor T is such that T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) match.
    # Actually, T is 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    
    def make_tensor(n, m, p):
        # Returns a list of (u, v, w) tuples where T is 1
        coords = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    coords.append((u, v, w))
        return coords

    coords_block = make_tensor(n_block, m_block, p_block)
    
    # ALS for rank R
    def als_decompose(coords, n_u, n_v, n_w, rank, max_iters=1000, tol=1e-10):
        # Initialize
        U = np.random.randn(n_u, rank)
        V = np.random.randn(n_v, rank)
        W = np.random.randn(n_w, rank)
        
        # Normalize columns
        for c in range(rank):
            U[:, c] /= np.linalg.norm(U[:, c])
            V[:, c] /= np.linalg.norm(V[:, c])
            W[:, c] /= np.linalg.norm(W[:, c])
            
        prev_err = np.inf
        for it in range(max_iters):
            # Update U
            # U[:, r] is the projection of the tensor slice onto V[:, r] and W[:, r]
            # T_u = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
            # This is expensive if done naively.
            # Since T is sparse (only 1s at coords), we can sum over coords.
            
            # For each r, we want U[:, r] such that sum_u U[u,r] T[u,v,w] V[v,r] W[w,r] is maximized?
            # No, ALS minimizes || T - sum U V^T W^T ||^2.
            # Fix V, W. Update U.
            # For each r, U[:, r] is the eigenvector corresponding to the largest eigenvalue of M_r,
            # where M_r[u, u'] = sum_{v,w} T[u,v,w] T[u',v,w] V[v,r] W[w,r] V[v',r] W[w',r]?
            # Actually, the objective is sum_{u,v,w} (T[u,v,w] - sum_r U[u,r]V[v,r]W[w,r])^2.
            # Expanding, the term for U is -2 sum_r U[u,r] (sum_{v,w} T[u,v,w] V[v,r] W[w,r]).
            # So U[:, r] is proportional to the vector b_r where b_r[u] = sum_{v,w} T[u,v,w] V[v,r] W[w,r].
            
            U_new = np.zeros_like(U)
            for r in range(rank):
                # b_r[u] = sum_{(u,v,w) in coords} T[u,v,w] V[v,r] W[w,r]
                # Since T is 1 at coords, b_r[u] = sum_{(u,v,w) in coords, u==u} V[v,r] W[w,r]
                # This is simply the sum of V[v,r]W[w,r] for all coords with that u.
                # We can precompute the structure.
                b = np.zeros(n_u)
                for (u, v, w) in coords:
                    b[u] += V[v, r] * W[w, r]
                U_new[:, r] = b
                norm = np.linalg.norm(U_new[:, r])
                if norm > 0:
                    U_new[:, r] /= norm
            
            # Update V
            V_new = np.zeros_like(V)
            for r in range(rank):
                b = np.zeros(n_v)
                for (u, v, w) in coords:
                    b[v] += U_new[u, r] * W[w, r]
                V_new[:, r] = b
                norm = np.linalg.norm(V_new[:, r])
                if norm > 0:
                    V_new[:, r] /= norm
            
            # Update W
            W_new = np.zeros_like(W)
            for r in range(rank):
                b = np.zeros(n_w)
                for (u, v, w) in coords:
                    b[w] += U_new[u, r] * V_new[v, r]
                W_new[:, r] = b
                norm = np.linalg.norm(W_new[:, r])
                if norm > 0:
                    W_new[:, r] /= norm
            
            U, V, W = U_new, V_new, W_new
            
            # Calculate error
            # err = || T - sum U V^T W^T ||^2
            # = sum_{coords} (1 - sum_r U[u,r]V[v,r]W[w,r])^2
            err = 0
            for (u, v, w) in coords:
                val = 0
                for r in range(rank):
                    val += U[u, r] * V[v, r] * W[w, r]
                err += (1 - val) ** 2
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    # Try to find rank 25 decomposition for 3x3x4
    best_U = None
    best_V = None
    best_W = None
    best_rank = 72 # fallback
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s for margin
    
    for target_rank in [25, 26, 27, 28]:
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt)
            U, V, W, err = als_decompose(coords_block, n_block*m_block, m_block*p_block, p_block*n_block, target_rank, max_iters=200, tol=1e-12)
            
            # Check if we can round to integers/half-integers
            # The values in U, V, W are scaled. We need to find a scaling factor such that U*scale, V*scale, W*scale are integers/half-integers?
            # Actually, the decomposition is unique up to scaling.
            # We can assume the entries are small integers.
            # Let's try to snap U, V, W to the nearest integer or half-integer.
            
            # First, normalize one vector to have max abs value 1?
            # Or just try to snap.
            
            U_snap = np.round(U * 2) / 2.0
            V_snap = np.round(V * 2) / 2.0
            W_snap = np.round(W * 2) / 2.0
            
            # Check error of snapped version
            err_snap = 0
            for (u, v, w) in coords_block:
                val = 0
                for r in range(target_rank):
                    val += U_snap[u, r] * V_snap[v, r] * W_snap[w, r]
                err_snap += (1 - val) ** 2
                
            if err_snap < 1e-6:
                # Success!
                best_U = U_snap
                best_V = V_snap
                best_W = W_snap
                best_rank = target_rank * 2 # For 3x3x8
                found = True
                break
                
        if found:
            break

    if best_U is not None:
        # Construct the 3x3x8 decomposition
        # U_full shape (9, best_rank)
        # V_full shape (24, best_rank)
        # W_full shape (24, best_rank)
        
        R_half = best_rank // 2
        
        U_full = np.zeros((9, best_rank))
        V_full = np.zeros((24, best_rank))
        W_full = np.zeros((24, best_rank))
        
        # Block 1: k in 0..3
        # U is the same
        U_full[:, :R_half] = best_U
        
        # V: rows 0..11 (j*4+k) for k=0..3
        # best_V has shape (12, R_half). Rows are j*4+k.
        # In V_full, rows are j*8+k.
        # So for j in 0..2, k in 0..3:
        # V_full[j*8+k, r] = best_V[j*4+k, r]
        for j in range(3):
            for k in range(4):
                V_full[j*8+k, :R_half] = best_V[j*4+k, :R_half]
                
        # W: rows 0..11 (k*3+i) for k=0..3
        # best_W has shape (12, R_half). Rows are k*3+i.
        # In W_full, rows are k*3+i.
        # So for k in 0..3, i in 0..2:
        # W_full[k*3+i, r] = best_W[k*3+i, r]
        for k in range(4):
            for i in range(3):
                W_full[k*3+i, :R_half] = best_W[k*3+i, :R_half]
        
        # Block 2: k in 4..7
        # U is the same
        U_full[:, R_half:] = best_U
        
        # V: rows 12..23 (j*8+k) for k=4..7
        # best_V has shape (12, R_half). Rows are j*4+k'. k' = k-4.
        # So V_full[j*8+k, r] = best_V[j*4+(k-4), r]
        for j in range(3):
            for k in range(4, 8):
                V_full[j*8+k, R_half:] = best_V[j*4+(k-4), R_half:]
                
        # W: rows 12..23 (k*3+i) for k=4..7
        # best_W has shape (12, R_half). Rows are k'*3+i. k' = k-4.
        # So W_full[k*3+i, r] = best_W[(k-4)*3+i, r]
        for k in range(4, 8):
            for i in range(3):
                W_full[k*3+i, R_half:] = best_W[(k-4)*3+i, R_half:]
        
        # Convert to lists
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
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
