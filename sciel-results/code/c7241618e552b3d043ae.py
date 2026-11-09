"""Matrix multiplication 4x4x6 tensor decomposition search."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96

    # Build schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = schoolbook_rank

    # Precompute index arrays for fast vectorized evaluation
    idx_i = np.repeat(np.arange(n), m*p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n*m)

    # Row indices for U, V, W
    U_rows = idx_i * m + idx_j
    V_rows = idx_j * p + idx_k
    W_rows = idx_k * n + idx_i

    def compute_error_vec(U, V, W):
        """Compute sum of squared errors using vectorized operations."""
        # For each triple (i,j,k), compute sum_c U[U_rows,c]*V[V_rows,c]*W[W_rows,c]
        # This is elementwise product then sum over columns
        # We need to gather the right rows
        U_gathered = U[U_rows]  # (nmp, rank)
        V_gathered = V[V_rows]  # (nmp, rank)
        W_gathered = W[W_rows]  # (nmp, rank)
        # Elementwise product, sum over rank
        vals = np.sum(U_gathered * V_gathered * W_gathered, axis=1)
        err = np.sum((1.0 - vals)**2)
        return err, vals

    def compute_gradients(U, V, W, vals):
        """Compute gradients of error w.r.t. U, V, W."""
        # err = sum_t (1 - vals[t])^2
        # d(err)/d(vals[t]) = -2*(1 - vals[t])
        # vals[t] = sum_c U[U_rows[t],c] * V[V_rows[t],c] * W[W_rows[t],c]
        # d(vals[t])/d(U[r,c]) = sum_{t: U_rows[t]==r} V[V_rows[t],c]*W[W_rows[t],c]
        # We use np.add.at for accumulation
        
        dvals = -2.0 * (1.0 - vals)  # (nmp,)
        
        U_gathered = U[U_rows]
        V_gathered = V[V_rows]
        W_gathered = W[W_rows]
        
        # d(err)/d(U[r,c]) = sum_t dvals[t] * V_gathered[t,c] * W_gathered[t,c]  (for t where U_rows[t]==r)
        dU_contrib = dvals[:, None] * V_gathered * W_gathered  # (nmp, rank)
        dU = np.zeros_like(U)
        np.add.at(dU, U_rows, dU_contrib)
        
        dV_contrib = dvals[:, None] * U_gathered * W_gathered
        dV = np.zeros_like(V)
        np.add.at(dV, V_rows, dV_contrib)
        
        dW_contrib = dvals[:, None] * U_gathered * V_gathered
        dW = np.zeros_like(W)
        np.add.at(dW, W_rows, dW_contrib)
        
        return dU, dV, dW

    def gradient_descent(rank, max_iters=5000, seed=42, lr_init=0.05):
        """Try to find a rank-r decomposition via gradient descent."""
        rng = np.random.RandomState(seed)
        
        # Initialize: start from schoolbook and randomly drop some columns,
        # or use random init
        if rank >= schoolbook_rank:
            U = U_sb[:, :rank].copy()
            V = V_sb[:, :rank].copy()
            W = W_sb[:, :rank].copy()
            # Add small random perturbation
            U += rng.randn(n*m, rank) * 0.01
            V += rng.randn(m*p, rank) * 0.01
            W += rng.randn(p*n, rank) * 0.01
        else:
            # Random initialization with moderate scale
            U = rng.randn(n*m, rank) * 0.5
            V = rng.randn(m*p, rank) * 0.5
            W = rng.randn(p*n, rank) * 0.5
        
        err, vals = compute_error_vec(U, V, W)
        best_err = err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            dU, dV, dW = compute_gradients(U, V, W, vals)
            
            # Normalize gradients to prevent explosion
            U_norm = np.linalg.norm(U)
            V_norm = np.linalg.norm(V)
            W_norm = np.linalg.norm(W)
            
            U -= lr * dU / max(U_norm, 1e-8) * U_norm
            V -= lr * dV / max(V_norm, 1e-8) * V_norm
            W -= lr * dW / max(W_norm, 1e-8) * W_norm
            
            new_err, new_vals = compute_error_vec(U, V, W)
            
            if new_err < err:
                err = new_err
                vals = new_vals
                lr = min(lr * 1.05, 0.5)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    break
            else:
                # Revert
                # We need to undo the update - simpler to just reduce lr and retry
                lr = max(lr * 0.7, 1e-6)
                # Recompute from best
                if err > best_err * 1.5:
                    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                    err, vals = compute_error_vec(U, V, W)
        
        if best_err < 1e-8:
            return best_U, best_V, best_W
        return None

    # Try various ranks with gradient descent
    for r in [95, 94, 93, 92, 91, 90, 88, 86, 84, 80, 76, 72]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            result = gradient_descent(r, max_iters=3000, seed=seed, lr_init=0.03)
            if result is not None:
                U_list, V_list, W_list = result
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = r
                    found = True
                    break
        if found:
            # Try to go even lower
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
