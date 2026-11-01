import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=5, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        for c in range(rank):
            A_flat = A.flatten()
            prod_A = A_flat * U_np[:, c]
            factor_A = prod_A.reshape(n, m).sum(axis=1)
            
            B_flat = B.flatten()
            prod_B = B_flat * V_np[:, c]
            factor_B = prod_B.reshape(m, p).sum(axis=0)
            
            W_mat = W_np[:, c].reshape(p, n)
            C_computed += np.outer(factor_A, factor_B) * W_mat.T
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _build_rank_56():
    """Build the baseline rank-56 decomposition using Strassen blocks."""
    n, m, p = DIMENSIONS
    U = np.zeros((n*m, 56))
    V = np.zeros((m*p, 56))
    W = np.zeros((p*n, 56))
    
    # Strassen coefficients for 2x2x2
    u_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0]
    ]
    
    v_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 0, 1]
    ]
    
    col_offset = 0
    for block_idx in range(8):
        is_left = block_idx < 4
        k_pair = block_idx % 4
        
        for s in range(7):
            if is_left:
                for i in range(2):
                    for j in range(2):
                        U[i*4 + j, col_offset] = u_strassen[s][i*2 + j]
                for j in range(2):
                    for k in range(2):
                        V[j*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                for i in range(2):
                    for k in range(2):
                        W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
            else:
                for i in range(2):
                    for j in range(2):
                        U[i*4 + (2 + j), col_offset] = u_strassen[s][i*2 + j]
                for j in range(2):
                    for k in range(2):
                        V[(2 + j)*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                for i in range(2):
                    for k in range(2):
                        W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
            
            col_offset += 1
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Start with Rank 56 Strassen
    try:
        U, V, W = _build_rank_56()
        
        # Global Column Removal Search
        # We attempt to remove columns one by one.
        # A column can be removed if its vector in the tensor space is in the span of the remaining columns.
        # Instead of full SVD on huge tensors, we can use a greedy approach with linear algebra checks.
        
        # To make this efficient, we can check linear dependence using a subset of random instances.
        # If a column is linearly dependent on others, removing it should still allow reconstruction.
        
        # A more robust check:
        # The tensor T is a sum of rank-1 tensors.
        # If column c is a linear combination of others, say v_c = sum lambda_i v_i,
        # then T = sum_{i != c} v_i u_i w_i^T + v_c u_c w_c^T
        # = sum_{i != c} (v_i + lambda_i v_c) u_i w_i^T + (v_c - v_c) ... wait.
        
        # Actually, we can just try to remove a column and verify if the decomposition still holds.
        # Verification is O(rank * n * m * p).
        # With rank ~50, this is fast.
        
        best_U, best_V, best_W = U, V, W
        best_rank = 56
        
        # Try to remove columns
        # We iterate through columns and try to remove them.
        # If removal fails, we keep it.
        # To allow for non-sequential dependencies, we might need multiple passes.
        
        max_passes = 5
        for _ in range(max_passes):
            if time.time() > deadline - 10:
                break
                
            current_rank = len(U[0])
            removed_any = False
            
            # Iterate backwards to handle index shifting
            for c in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Try removing column c
                U_new = np.delete(U, c, axis=1)
                V_new = np.delete(V, c, axis=1)
                W_new = np.delete(W, c, axis=1)
                
                if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, trials=2, tol=1e-9):
                    # Success!
                    U, V, W = U_new, V_new, W_new
                    best_U, best_V, best_W = U, V, W
                    best_rank = len(U[0])
                    removed_any = True
            
            if not removed_any:
                break
        
        if best_rank < 56:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}
