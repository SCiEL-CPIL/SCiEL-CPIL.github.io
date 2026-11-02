import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[a, b, c]."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T_target):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_target[a,b,c] for all a,b,c."""
    rank = U.shape[1]
    T_decomp = np.zeros(T_target.shape, dtype=np.int64)
    for r in range(rank):
        u_col = U[:, r].astype(np.int64)
        v_col = V[:, r].astype(np.int64)
        w_col = W[:, r].astype(np.int64)
        nonzero_u = np.nonzero(u_col)[0]
        if len(nonzero_u) == 0:
            continue
        nonzero_v = np.nonzero(v_col)[0]
        if len(nonzero_v) == 0:
            continue
        nonzero_w = np.nonzero(w_col)[0]
        if len(nonzero_w) == 0:
            continue
        for a in nonzero_u:
            ua = u_col[a]
            for b in nonzero_v:
                vb = v_col[b]
                uv = ua * vb
                if uv == 0:
                    continue
                for c in nonzero_w:
                    wc = w_col[c]
                    T_decomp[a, b, c] += uv * wc
    return np.array_equal(T_decomp, T_target)

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
    return U, V, W

def _to_half_integers(U, V, W, tol=1e-8):
    """
    Try to convert float factor matrices to half-integer representations.
    Returns (U_hi, V_hi, W_hi) as int64 arrays representing 2*value, or None if fails.
    """
    # Find a common scaling: try to express all entries as multiples of 0.5
    # First, check if all entries are close to half-integers
    def check_half_int(arr, tol):
        doubled = arr * 2
        rounded = np.round(doubled)
        max_err = np.max(np.abs(doubled - rounded))
        return max_err < tol, rounded.astype(np.int64)
    
    ok_u, U_hi = check_half_int(U, tol)
    ok_v, V_hi = check_half_int(V, tol)
    ok_w, W_hi = check_half_int(W, tol)
    
    if ok_u and ok_v and ok_w:
        return U_hi, V_hi, W_hi
    return None

def _try_structured_rank(rank, n, m, p, max_iters=200, seed=0):
    """
    Try to find a rank-`rank` decomposition using structured random initialization
    and alternating least squares on the tensor.
    """
    rng = np.random.RandomState(seed)
    T = _build_target_tensor(n, m, p)
    
    # Initialize with random factors
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Normalize columns to prevent scale drift
    for r in range(rank):
        norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
        if norm > 0:
            scale = norm ** (1.0 / 3.0)
            U[:, r] /= scale
            V[:, r] /= scale
            W[:, r] /= scale
    
    for it in range(max_iters):
        # Update U: for each a, solve min sum_b,c T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to: for each a, the vector of T[a,:,:] should equal sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
        # So we can do least squares: T[a,:,:].flatten() = U[a,:] @ M where M[r, (b,c)] = V[b,r]*W[c,r]
        
        # Build M: shape (rank, m*p * p*n)
        M = np.zeros((rank, m * p * p * n))
        for r in range(rank):
            M[r, :] = (V[:, r] * W[:, r]).flatten()  # V[:,r] is (m*p,), W[:,r] is (p*n,)
            # Actually V[:,r] has length m*p, W[:,r] has length p*n
            # Outer product V[:,r] ⊗ W[:,r] has shape (m*p, p*n)
            # Flatten gives length m*p*p*n
        
        # T_flat[a] = T[a,:,:].flatten() has length m*p*p*n
        T_flat = T.reshape(n * m, -1)
        
        # Solve T_flat = U @ M  =>  U = T_flat @ M^T @ (M M^T)^{-1}
        # But this is overdetermined if rank > n*m. Use least squares.
        try:
            # U = argmin ||T_flat - U @ M||_F
            # This is a linear least squares problem: for each column of T_flat
            # Actually: T_flat[a, :] = sum_r U[a,r] * M[r, :]
            # So: T_flat = U @ M
            # Solve for U: U = T_flat @ M^T @ (M @ M^T)^{-1} if rank(M) = rank
            MMT = M @ M.T
            # Add small regularization
            MMT += 1e-12 * np.eye(rank)
            U_new = T_flat @ M.T @ np.linalg.solve(MMT, np.eye(rank))
            U = U_new
        except np.linalg.LinAlgError:
            pass
        
        # Update V
        # T_flat2[b, :] = T[:,b,:].reshape(n*m, -1).T.flatten() ... 
        # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed b: T[:,b,:] has shape (n*m, p*n)
        # T[:,b,:].flatten() = V[b,:] @ N where N[r, (a,c)] = U[a,r]*W[c,r]
        
        N = np.zeros((rank, n * m * p * n))
        for r in range(rank):
            N[r, :] = (U[:, r] * W[:, r]).flatten()
        
        T_flat_b = T.reshape(n * m, m * p, p * n).transpose(1, 0, 2).reshape(m * p, -1)
        
        try:
            NNT = N @ N.T
            NNT += 1e-12 * np.eye(rank)
            V_new = T_flat_b @ N.T @ np.linalg.solve(NNT, np.eye(rank))
            V = V_new
        except np.linalg.LinAlgError:
            pass
        
        # Update W
        # For fixed c: T[:,:,c] has shape (n*m, m*p)
        # T[:,:,c].flatten() = W[c,:] @ P where P[r, (a,b)] = U[a,r]*V[b,r]
        
        P = np.zeros((rank, n * m * m * p))
        for r in range(rank):
            P[r, :] = (U[:, r] * V[:, r]).flatten()
        
        T_flat_c = T.reshape(n * m, m * p, p * n).transpose(2, 0, 1).reshape(p * n, -1)
        
        try:
            PPT = P @ P.T
            PPT += 1e-12 * np.eye(rank)
            W_new = T_flat_c @ P.T @ np.linalg.solve(PPT, np.eye(rank))
            W = W_new
        except np.linalg.LinAlgError:
            pass
        
        # Normalize
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                scale = norm ** (1.0 / 3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
    
    # Verify
    T_decomp = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n * m, m * p, p * n)
    
    if np.allclose(T_decomp, T, atol=1e-6):
        # Try to convert to half-integers
        result = _to_half_integers(U, V, W, tol=1e-5)
        if result is not None:
            return result
    return None

def _try_rank_random(rank, n, m, p, max_iters=5000, seed=42):
    """
    Try random search for a rank-`rank` decomposition.
    """
    rng = np.random.RandomState(seed)
    T = _build_target_tensor(n, m, p)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize
    for r in range(rank):
        norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
        if norm > 0:
            scale = norm ** (1.0 / 3.0)
            U[:, r] /= scale
            V[:, r] /= scale
            W[:, r] /= scale
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Gradient descent step
        # Compute error tensor
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n * m, m * p, p * n)
        
        err = T_decomp - T
        err_flat = err.reshape(n * m, m * p, p * n)
        
        # Gradient w.r.t. U[a,r] = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            # grad_U[:, r] = err @ (V[:,r] ⊗ W[:,r])
            # err has shape (n*m, m*p, p*n)
            # For each a: grad_U[a,r] = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
            # This is: err.reshape(n*m, m*p*p*n) @ (V[:,r]*W[:,r]).flatten()
            vw = (V[:, r] * W[:, r]).flatten()
            grad_U[:, r] = err.reshape(n * m, -1) @ vw
            
            uv = (U[:, r] * W[:, r]).flatten()
            grad_V[:, r] = err.reshape(-1, m * p, p * n).transpose(1, 0, 2).reshape(m * p, -1) @ uv
            
            uv = (U[:, r] * V[:, r]).flatten()
            grad_W[:, r] = err.reshape(n * m, m * p, -1).transpose(2, 0, 1).reshape(p * n, -1) @ uv
        
        # Update
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Normalize
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                scale = norm ** (1.0 / 3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # Check error
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n * m, m * p, p * n)
        
        err_val = np.linalg.norm(T_decomp - T)
        if err_val < best_err:
            best_err = err_val
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-5:
        result = _to_half_integers(best_U, best_V, best_W, tol=1e-4)
        if result is not None:
            return result
    return None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 63).
    2. Try to find lower rank decompositions using structured methods.
    3. Return the best verified decomposition found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 20s buffer before 120s limit

    # Build target tensor for verification
    T_target = _build_target_tensor(n, m, p)

    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Strassen-like methods might achieve rank between 21 and 63
    
    # Try structured approach for various ranks
    for rank in range(21, min(63, best_rank)):
        if time.time() > deadline - 30:
            break
        
        # Try structured random initialization
        result = _try_structured_rank(rank, n, m, p, max_iters=100, seed=rank)
        if result is not None:
            U_hi, V_hi, W_hi = result
            # Verify
            if _verify_decomposition(U_hi, V_hi, W_hi, T_target):
                best_rank = rank
                best_U, best_V, best_W = U_hi, V_hi, W_hi
                break  # Found the lowest possible rank for this approach
    
    # Try random search for a few more ranks if time permits
    if time.time() < deadline - 20:
        for rank in range(21, min(30, best_rank)):
            if time.time() > deadline - 10:
                break
            result = _try_rank_random(rank, n, m, p, max_iters=2000, seed=rank + 100)
            if result is not None:
                U_hi, V_hi, W_hi = result
                if _verify_decomposition(U_hi, V_hi, W_hi, T_target):
                    best_rank = rank
                    best_U, best_V, best_W = U_hi, V_hi, W_hi
                    break

    # Final verification
    if time.time() < deadline - 2:
        if not _verify_decomposition(best_U, best_V, best_W, T_target):
            # Fallback: rebuild schoolbook
            best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)

    # Convert to lists for JSON serialization
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
