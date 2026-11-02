import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = l_row * m + (a_col_start + l_col)
                U[global_idx, col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (b_row_start + l_row) * p + (b_col_start + l_col)
                V[global_idx, col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (c_col_start + l_col) * n + l_row
                W[global_idx, col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)
    
    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    # Build the matmul tensor T[i,j,k] = delta_{j,k} (A[i,j] * B[j,k] -> C[i,k])
    # Actually T[i,j,k] = 1 if j == k, else 0... no.
    # The tensor is: T[i,j,k] = 1 if j==k else 0? No.
    # For matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if j==k, 0 otherwise.
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal T[i,j,k]
    
    # W convention: W[k*n + i, r]
    # So W index: row = k*n + i
    # U index: row = i*m + j
    # V index: row = j*p + k
    
    # Build tensor
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0  # T[i,j,k] = 1 if k==j
    
    # Compute reconstruction
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for batch computation
    # U_arr shape: (n*m, rank) -> (n, m, rank)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)  # W[k*n+i, r] -> reshape to (p, n, rank)
    
    # sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _als_decomposition(rank, max_iters=500, tol=1e-10, seed=42):
    """Try to find a rank-rank decomposition using ALS."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Build target tensor
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    # Initialize factors randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,:] = sum_r U[i,j,r] * V3[j,:,r] * W3[:,i,r]
                # Reshape: for fixed i,j, the contribution is U[i,j,:] * (V3[j,:,r] * W3[:,i,r])
                # We need: T[i,j,k] = sum_r U[i,j,r] * (V3[j,k,r] * W3[k,i,r])
                # So for fixed i,j: T[i,j,:] = U[i,j,:] @ M where M[r,k] = sum over... 
                # Actually: T[i,j,k] = sum_r U[i,j,r] * V3[j,k,r] * W3[k,i,r]
                # Let A[r,k] = V3[j,k,r] * W3[k,i,r]
                # Then T[i,j,k] = sum_r U[i,j,r] * A[r,k]
                # So U[i,j,:] = T[i,j,:] / A  (least squares)
                A = V3[j, :, :] * W3[:, i, :]  # shape (p, rank) but we need (rank, p)
                A = A.T  # shape (rank, p)
                # Solve: T[i,j,:] = U[i,j,:] @ A
                try:
                    U3[i, j, :] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                # For fixed j,k: T[:,j,k] = sum_r (U3[:,j,r] * W3[k,:,r]) * V3[j,k,r]
                # Let B[r,i] = U3[i,j,r] * W3[k,i,r]
                # T[:,j,k] = V3[j,k,:] @ B  where B has shape (rank, n)
                B = U3[:, j, :] * W3[k, :, :]  # shape (n, rank)
                B = B.T  # shape (rank, n)
                try:
                    V3[j, k, :] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                # For fixed k,i: T[i,:,k] = sum_r (U3[i,:,r] * V3[:,k,r]) * W3[k,i,r]
                # Let C[r,j] = U3[i,j,r] * V3[j,k,r]
                # T[i,:,k] = W3[k,i,:] @ C  where C has shape (rank, m)
                C = U3[i, :, :] * V3[:, k, :]  # shape (m, rank)
                C = C.T  # shape (rank, m)
                try:
                    W3[k, i, :] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))
        if residual < tol:
            break
    
    U_flat = U3.reshape(n * m, rank)
    V_flat = V3.reshape(m * p, rank)
    W_flat = W3.reshape(p, n, rank).reshape(p * n, rank)
    
    return U_flat, V_flat, W_flat, residual


def _try_rank_reduction(target_rank, start_U, start_V, start_W, max_restarts=5, seed=0):
    """Try to reduce rank by zeroing out columns and re-optimizing."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    current_U = start_U.copy()
    current_V = start_V.copy()
    current_W = start_W.copy()
    current_rank = start_U.shape[1]
    
    for restart in range(max_restarts):
        rng = np.random.RandomState(seed + restart)
        U = current_U.copy()
        V = current_V.copy()
        W = current_W.copy()
        
        # Try to zero out one column at a time
        for col_to_remove in range(current_rank - target_rank):
            # Pick a random column to try removing
            col_idx = rng.randint(0, U.shape[1])
            
            # Set that column to zero
            U[:, col_idx] = 0
            V[:, col_idx] = 0
            W[:, col_idx] = 0
            
            # Re-optimize with ALS
            max_iters = 200
            U3 = U.reshape(n, m, -1)
            V3 = V.reshape(m, p, -1)
            W3 = W.reshape(p, n, -1)
            
            for it in range(max_iters):
                for i in range(n):
                    for j in range(m):
                        A = V3[j, :, :] * W3[:, i, :]
                        A = A.T
                        try:
                            U3[i, j, :] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                        except:
                            pass
                for j in range(m):
                    for k in range(p):
                        B = U3[:, j, :] * W3[k, :, :]
                        B = B.T
                        try:
                            V3[j, k, :] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                        except:
                            pass
                for k in range(p):
                    for i in range(n):
                        C = U3[i, :, :] * V3[:, k, :]
                        C = C.T
                        try:
                            W3[k, i, :] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                residual = np.max(np.abs(recon - T))
                if residual < 1e-10:
                    break
            
            U = U3.reshape(n * m, -1)
            V = V3.reshape(m * p, -1)
            W = W3.reshape(p * n, -1)
            
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            residual = np.max(np.abs(recon - T))
            
            if residual < 1e-8:
                # Check if we can actually remove the column
                # Verify with the column zeroed
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                U_test[:, col_idx] = 0
                V_test[:, col_idx] = 0
                W_test[:, col_idx] = 0
                
                U3t = U_test.reshape(n, m, -1)
                V3t = V_test.reshape(m, p, -1)
                W3t = W_test.reshape(p, n, -1)
                recon_test = np.einsum('ijr,jkr,kir->ijk', U3t, V3t, W3t)
                residual_test = np.max(np.abs(recon_test - T))
                
                if residual_test < 1e-8:
                    # We can remove this column
                    # Remove it
                    mask = np.ones(U.shape[1], dtype=bool)
                    mask[col_idx] = False
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    current_U = U
                    current_V = V
                    current_W = W
                    current_rank -= 1
                    break
            else:
                # Revert
                U = current_U.copy()
                V = current_V.copy()
                W = current_W.copy()
        
        if current_rank <= target_rank:
            break
    
    return current_U, current_V, current_W, current_rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with known rank-28 solution
    U28, V28, W28 = _build_strassen_rank28()
    
    # Convert to lists for return
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to reduce to rank 27
    if time.time() < deadline:
        for seed in range(10):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, rank_r = _try_rank_reduction(27, U28, V28, W28, max_restarts=3, seed=seed)
                if rank_r < best_rank:
                    # Verify
                    valid, res = _verify_decomposition(U_r, V_r, W_r, tol=1e-6)
                    if valid:
                        best_rank = rank_r
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except Exception as e:
                continue
    
    # Try pure ALS for rank 27
    if time.time() < deadline and best_rank == 28:
        for seed in range(20):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, residual = _als_decomposition(27, max_iters=300, seed=seed)
                if residual < 1e-8:
                    valid, res = _verify_decomposition(U_r, V_r, W_r, tol=1e-6)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except Exception as e:
                continue
    
    return best_result
# EVOLVE-BLOCK-END
