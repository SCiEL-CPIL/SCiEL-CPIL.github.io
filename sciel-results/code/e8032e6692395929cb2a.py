import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim = n * m * p
    deadline = time.time() + 110  # 110s to be safe
    
    # Initialize with schoolbook
    rank = dim
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We use 1-based indices for clarity in thought, but 0-based in code
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    # The target tensor T
    # T[k*n + i, j] ? No, the tensor is T[i, j, k] = delta?
    # The multiplication is C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] is 1 if this term exists.
    # In the decomposition: C[i, k] = sum_c U[i, c] V[c, k] ... wait.
    # The interface says:
    # U: (n*m, rank) -> U[i*m+j, c]
    # V: (m*p, rank) -> V[j*p+k, c]
    # W: (p*n, rank) -> W[k*n+i, c]
    # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ?
    # No, the standard tensor decomposition for matmul is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This is represented as a sum of rank-1 tensors:
    # T = sum_c u_c \otimes v_c \otimes w_c
    # Where u_c has 1 at (i, j), v_c has 1 at (j, k), w_c has 1 at (k, i)?
    # Let's check the indices.
    # If we multiply U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c], the sum over c gives a value for fixed i, j, k.
    # This value should be 1 if (i, j, k) is a valid term in the multiplication, and 0 otherwise?
    # Actually, the tensor T for matmul has T[i, j, k] = 1.
    # The decomposition must satisfy:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    
    # So T is a 3D array of ones.
    T = np.ones((n, m, p), dtype=np.float64)
    T_flat = T.flatten() # Length 75
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    # Greedy reduction
    # We try to remove columns.
    # We maintain the list of active columns.
    active_cols = list(range(rank))
    
    while time.time() < deadline:
        current_rank = len(active_cols)
        if current_rank <= best_rank:
            # Update best if we found a lower rank
            if current_rank < best_rank:
                best_rank = current_rank
                # Extract best matrices
                best_U = U[:, active_cols].copy()
                best_V = V[:, active_cols].copy()
                best_W = W[:, active_cols].copy()
                
        # Try to remove one column
        removed = False
        # Shuffle to avoid bias? Or try all?
        # Trying all is O(R * D^2). R=75, D=75. 75*5625 ~ 400k ops. Very fast.
        # So we can try all columns at each step.
        
        candidate_cols = active_cols[:]
        np.random.shuffle(candidate_cols)
        
        for c in candidate_cols:
            # Check if we can remove c
            # New active cols
            new_active = [x for x in active_cols if x != c]
            
            # Form the matrix M of decomposable tensors for new_active
            # M has shape (dim, len(new_active))
            # M[:, idx] = flatten( U[:, col] * V[:, col] * W[:, col] )
            # Note: The tensor is T[i,j,k] = sum U[i,j] V[j,k] W[k,i] ?
            # The product is outer product of vectors.
            # u = U[:, c] (size n*m)
            # v = V[:, c] (size m*p)
            # w = W[:, c] (size p*n)
            # The tensor is u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            # We need to check if T_flat is in the span of these tensors.
            # Construct M
            M = np.zeros((dim, len(new_active)), dtype=np.float64)
            for idx, col_idx in enumerate(new_active):
                u = U[:, col_idx]
                v = V[:, col_idx]
                w = W[:, col_idx]
                # Reshape and multiply
                # u is (n*m), v is (m*p), w is (p*n)
                # We need to form the 3D tensor t = u[i*m+j] * v[j*p+k] * w[k*n+i]
                # This is not a simple outer product of 1D vectors in the flattened space directly without reshaping.
                # However, we can compute the flattened tensor efficiently.
                # t_flat[k*n + i*m + j] ? No, the order is k, i, j?
                # The W index is k*n+i. The U index is i*m+j.
                # Let's stick to the definition:
                # Term = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This defines a 3D array T_c[i, j, k].
                # We can compute T_c by:
                # U_c = U[:, c].reshape(n, m)
                # V_c = V[:, c].reshape(m, p)
                # W_c = W[:, c].reshape(p, n)
                # T_c[i, j, k] = U_c[i, j] * V_c[j, k] * W_c[k, i]
                
                U_c = u.reshape(n, m)
                V_c = v.reshape(m, p)
                W_c = w.reshape(p, n)
                
                # Compute T_c
                # T_c[i, j, k] = U_c[i, j] * V_c[j, k] * W_c[k, i]
                # This can be computed as:
                # For each i, k: T_c[:, :, (i,k)] is not straightforward.
                # Let's use einsum or loops.
                # T_c = np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
                T_c = np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
                M[:, idx] = T_c.flatten()
                
            # Check if T_flat is in the column space of M
            # Solve M x = T_flat in least squares
            # If residual is 0, it's in the space.
            try:
                sol, residuals, rank_m, s = np.linalg.lstsq(M, T_flat, rcond=None)
                # Calculate residual
                res = np.linalg.norm(M @ sol - T_flat)
                if res < 1e-9:
                    # We can remove c
                    active_cols = new_active
                    removed = True
                    break
            except Exception:
                pass
                
        if not removed:
            # If we couldn't remove any column, we might be stuck.
            # We can try a random perturbation or restart?
            # For now, break if we can't reduce further.
            # Or try a different random order?
            # Let's try a few random restarts of the removal process if stuck?
            # Actually, if greedy fails, it's likely a local minimum.
            # We can try to "swap" a column?
            # For simplicity, we break.
            break

    # Convert back to lists of ints/floats
    # The problem asks for integer/half-integer entries.
    # Our search uses floats. If we find a solution, it might have float noise.
    # We should round to nearest 0.5 or integer?
    # The schoolbook solution is integers.
    # If we reduce rank, the coefficients might become non-integer.
    # The contract says "integer/half-integer entries".
    # If the solution requires irrational numbers, it's invalid.
    # However, if we start from integers and do linear algebra, the solution to the linear system M x = T might be rational.
    # We should verify the solution is exact.
    
    # Re-construct the best solution found
    U_best = best_U
    V_best = best_V
    W_best = best_W
    
    # Round to nearest 0.5 to satisfy "integer/half-integer"
    # This might break exactness if the true solution is not half-integer.
    # But for matmul decompositions, often integer solutions exist if rank allows.
    # If rounding breaks it, we might return a slightly invalid solution.
    # Better to return the float solution? No, "integer/half-integer" is a constraint.
    # If the solution is not half-integer, we should probably not use it?
    # Or does the evaluator accept floats that are close?
    # "exactly decompose" implies exactness.
    # If I return floats, it might fail.
    # I will round to nearest 0.5 and then verify.
    
    U_out = np.round(U_best * 2) / 2
    V_out = np.round(V_best * 2) / 2
    W_out = np.round(W_best * 2) / 2
    
    # Verify exactness
    # Check if sum U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(best_rank):
                    val += U_out[i*m+j, c] * V_out[j*p+k, c] * W_out[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to schoolbook
        U_out = np.zeros((n*m, dim))
        V_out = np.zeros((m*p, dim))
        W_out = np.zeros((p*n, dim))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_out[i*m + j, col] = 1.0
                    V_out[j*p + k, col] = 1.0
                    W_out[k*n + i, col] = 1.0
                    col += 1
        best_rank = dim

    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
