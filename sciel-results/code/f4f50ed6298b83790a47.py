import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U @ V @ W reconstruction matches target tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_rec = np.zeros(T.shape)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_rec, T, atol=tol)

def _round_to_halfint(arr, tol=1e-3):
    """Round array to nearest half-integers, return (rounded, max_error)."""
    rounded = np.round(arr * 2.0) / 2.0
    max_err = np.max(np.abs(arr - rounded)) if arr.size > 0 else 0.0
    return rounded, max_err

def _als_decompose(T, n, m, p, rank, max_iters=80, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    """
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    prev_loss = np.inf
    for it in range(max_iters):
        # Solve for U
        X = np.zeros((rank, m * p * p * n))
        for r in range(rank):
            X[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        T_flat = T.reshape(n * m, m * p * p * n)
        U_new, _, _, _ = np.linalg.lstsq(X.T, T_flat.T, rcond=None)
        U = U_new.T
        
        # Solve for V
        Y = np.zeros((rank, n * m * p * n))
        for r in range(rank):
            Y[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        T_stacked = np.zeros((m * p, n * m * p * n))
        for v in range(m * p):
            T_stacked[v, :] = T[:, v, :].flatten()
        V_new, _, _, _ = np.linalg.lstsq(Y.T, T_stacked.T, rcond=None)
        V = V_new.T
        
        # Solve for W
        Z = np.zeros((rank, n * m * m * p))
        for r in range(rank):
            Z[r, :] = np.outer(U[:, r], V[:, r]).flatten()
        T_stacked_w = np.zeros((p * n, n * m * m * p))
        for w in range(p * n):
            T_stacked_w[w, :] = T[:, :, w].flatten()
        W_new, _, _, _ = np.linalg.lstsq(Z.T, T_stacked_w.T, rcond=None)
        W = W_new.T
        
        # Calculate loss
        T_rec = np.zeros(T.shape)
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        loss = np.sum((T_rec - T) ** 2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
    return U, V, W, loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    T = _build_target_tensor(n, m, p)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    # Try to find lower rank decompositions using ALS
    # Known results suggest rank might be reducible for some matrix multiplication sizes
    for target_rank in range(44, 20, -1):
        if time.time() > deadline - 25:
            break
            
        found = False
        n_restarts = 3
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(1000 + restart * 7 + target_rank)
            
            try:
                U, V, W, loss = _als_decompose(T, n, m, p, target_rank, max_iters=60)
                
                if loss < 1e-6:
                    # Round to half-integers
                    U_rounded, err_u = _round_to_halfint(U)
                    V_rounded, err_v = _round_to_halfint(V)
                    W_rounded, err_w = _round_to_halfint(W)
                    
                    max_err = max(err_u, err_v, err_w)
                    
                    if max_err < 1e-4:
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                            best_result = {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            best_rank = target_rank
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            continue

    return best_result
