"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and compare with the multiplication tensor (which is 1 for all valid (i,j,k)).
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification using numpy."""
    rank = U.shape[1]
    # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We can do this via: for each c, outer product contribution
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Create 4D tensor: T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Sum over c.
    # Efficient: use einsum or manual loops over c
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # (n*m,)
        vc = V[:, c]  # (m*p,)
        wc = W[:, c]  # (p*n,)
        # uc reshaped to (n,m), vc to (m,p), wc to (p,n)
        uc_2d = uc.reshape(n, m)
        vc_2d = vc.reshape(m, p)
        wc_2d = wc.reshape(p, n)
        # T[i,j,k] += uc_2d[i,j] * vc_2d[j,k] * wc_2d[k,i]
        # This is like: for each i,j,k, product of three elements
        # We can compute via: sum over j for fixed i,k? No, all three are independent indices.
        # Actually: T[i,j,k] = uc_2d[i,j] * vc_2d[j,k] * wc_2d[k,i]
        # This is a rank-1-like update but with shared indices.
        # Let's just do it directly for small sizes.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc_2d[i, j] * vc_2d[j, k] * wc_2d[k, i]
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol


def _rank_test(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns True if valid."""
    return _verify_decomposition_fast(U, V, W, n, m, p, tol)


def _local_optimization(U, V, W, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """
    Local optimization to reduce error. Uses alternating least squares / gradient steps.
    We want T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    def compute_error():
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        return np.sum((T - 1.0)**2)
    
    best_err = compute_error()
    if best_err < tol:
        return U, V, W, best_err
    
    lr = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        residual = T - 1.0
        
        # Gradient w.r.t. U[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c] summed over k
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = residual[i, j, k]
                        grad_U[i * m + j, c] += r * vc[j, k] * wc[k, i]
                        grad_V[j * p + k, c] += r * uc[i, j] * wc[k, i]
                        grad_W[k * n + i, c] += r * uc[i, j] * vc[j, k]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        err = compute_error()
        if err < best_err:
            best_err = err
        if best_err < tol:
            break
        # Adaptive learning rate
        if err > best_err * 1.1:
            lr *= 0.5
        elif err < best_err * 0.9:
            lr = min(lr * 1.2, 0.1)
    
    return U, V, W, best_err


def _try_rank(rank, n, m, p, deadline, num_restarts=3):
    """Try to find a valid decomposition of given rank."""
    best_U, best_V, best_W, best_err = None, None, None, float('inf')
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        # Random initialization
        np.random.seed(42 + restart)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Scale to make initial error reasonable
        U, V, W, err = _local_optimization(U, V, W, n, m, p, max_iter=100, tol=1e-10, deadline=deadline)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-10:
            break
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try lower ranks in increasing order
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But tensor rank can be lower than these in some cases... actually for matrix mult,
    # rank >= max(n*m, m*p, p*n) is NOT always true, but it's a good heuristic lower bound.
    # Actually the true lower bound is more subtle. Let's try from 15 upward.
    
    for rank in range(15, best_rank):
        if time.time() > deadline - 20:
            break
        result = _try_rank(rank, n, m, p, deadline, num_restarts=2)
        if result is not None:
            U, V, W = result
            if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid lower rank, stop
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
