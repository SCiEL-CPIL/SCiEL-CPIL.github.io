import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # Safety margin

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] for matrix multiplication."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) > tol:
        return False
    # Check integer/half-integer constraint
    # The problem states "integer/half-integer entries"
    # This usually means entries must be in 0.5 * Z
    U_h = U * 2
    V_h = V * 2
    W_h = W * 2
    if not np.allclose(U_h, np.round(U_h), atol=1e-5):
        return False
    if not np.allclose(V_h, np.round(V_h), atol=1e-5):
        return False
    if not np.allclose(W_h, np.round(W_h), atol=1e-5):
        return False
    return True

def _als_decompose(T, R, n_iter=200, seed=42, tol=1e-6):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(A, R) * 0.1
    V = rng.randn(B, R) * 0.1
    W = rng.randn(C, R) * 0.1
    
    # Precompute target slices for efficiency
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 5:
            return None
            
        # Solve for U
        # T_a[a] = U[a, :] * (V * W)_{(b,c)}
        # Let X = V.flatten() * W.flatten() reshaped to (B*C, R)
        A_U = np.outer(V.flatten(), W.flatten()).reshape(B * C, R)
        for a in range(A):
            U[a, :], _, _, _ = np.linalg.lstsq(A_U, T_a[a], rcond=None)
        
        # Solve for V
        A_V = np.outer(U.flatten(), W.flatten()).reshape(A * C, R)
        for b in range(B):
            V[b, :], _, _, _ = np.linalg.lstsq(A_V, T_b[b], rcond=None)
        
        # Solve for W
        A_W = np.outer(U.flatten(), V.flatten()).reshape(A * B, R)
        for c in range(C):
            W[c, :], _, _, _ = np.linalg.lstsq(A_W, T_c[c], rcond=None)
            
        # Check convergence
        if it % 10 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-5:
        return U, V, W
    return None

def _try_round_and_verify(U, V, W, T):
    """
    Try to round U, V, W to half-integers and verify.
    If direct rounding fails, try a local search around the rounded values.
    """
    # Scale by 2 to work with integers
    U2 = U * 2
    V2 = V * 2
    W2 = W * 2
    
    # Simple rounding
    U_r = np.round(U2)
    V_r = np.round(V2)
    W_r = np.round(W2)
    
    # Verify
    if _verify(U_r/2, V_r/2, W_r/2, T):
        return (U_r/2).tolist(), (V_r/2).tolist(), (W_r/2).tolist()
    
    # If simple rounding fails, try a small grid search around the rounded values
    # This is expensive, so we only do it if the error is very small
    T_recon = np.einsum('ar,br,cr->abc', U_r/2, V_r/2, W_r/2)
    err = np.max(np.abs(T_recon - T))
    
    if err < 1.0: # Only if we are somewhat close
        # Try flipping signs or adjusting by 1 for a few random entries
        # This is a heuristic
        for _ in range(20):
            if time.time() > _DEADLINE - 2:
                break
            U_t = U_r.copy()
            V_t = V_r.copy()
            W_t = W_r.copy()
            
            # Perturb a few random entries
            for _ in range(3):
                mat = np.random.choice([U_t, V_t, W_t])
                idx = np.random.randint(mat.size)
                r, c = np.unravel_index(idx, mat.shape)
                mat[r, c] += np.random.choice([-1, 1])
            
            if _verify(U_t/2, V_t/2, W_t/2, T):
                return (U_t/2).tolist(), (V_t/2).tolist(), (W_t/2).tolist()
                
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Theoretical lower bound is around 20.
    # We try ranks 20, 21, 22, 23.
    # If we find one, we return it.
    # Otherwise, we return schoolbook (rank 24).
    
    for rank in [20, 21, 22, 23]:
        if time.time() > _DEADLINE - 20:
            break
            
        # Number of attempts per rank
        # We spend more time on lower ranks if they are promising, but here we just try a fixed number.
        num_attempts = 100 if rank <= 21 else 50
        
        for seed in range(num_attempts):
            if time.time() > _DEADLINE - 10:
                break
                
            res = _als_decompose(T, rank, n_iter=100, seed=seed)
            if res is not None:
                U, V, W = res
                rounded_res = _try_round_and_verify(U, V, W, T)
                if rounded_res is not None:
                    return {"U": rounded_res[0], "V": rounded_res[1], "W": rounded_res[2]}
                    
    # Fallback: Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
