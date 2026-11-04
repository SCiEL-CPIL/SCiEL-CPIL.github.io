import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 115.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen decompositions. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    r = r2 * r2  # 49
    
    U2_3d = U2.reshape(2, 2, r2)
    V2_3d = V2.reshape(2, 2, r2)
    W2_3d = W2.reshape(2, 2, r2)
    
    U4_5d = np.einsum('abk,cdl->acbdkl', U2_3d, U2_3d)
    U4 = U4_5d.reshape(16, 49).astype(np.int16)
    V4 = np.einsum('abk,cdl->acbdkl', V2_3d, V2_3d).reshape(16, 49).astype(np.int16)
    W4 = np.einsum('abk,cdl->acbdkl', W2_3d, W2_3d).reshape(16, 49).astype(np.int16)
    
    return U4, V4, W4

def _build_4x4x8():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98."""
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def _verify_fast(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Faster verification using vectorized operations."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    
    # Vectorized computation
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def _try_optimize(U, V, W, n=4, m=4, p=8, time_budget=60.0):
    """
    Attempt to reduce rank by optimizing and zeroing small entries.
    Uses L-BFGS-B to minimize residual, then iteratively removes small components.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Convert to float for optimization
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    current_rank = U_f.shape[1]
    
    # Try to reduce rank by zeroing out small entries
    # First, let's try a simple approach: remove columns with small norm
    for attempt in range(10):
        if time.time() - start_time > time_budget:
            break
        
        # Compute column norms
        U_norms = np.linalg.norm(U_f, axis=0)
        V_norms = np.linalg.norm(V_f, axis=0)
        W_norms = np.linalg.norm(W_f, axis=0)
        
        # Find the smallest column
        min_U_idx = np.argmin(U_norms)
        min_V_idx = np.argmin(V_norms)
        min_W_idx = np.argmin(W_norms)
        
        # Try removing the smallest one
        candidates = [
            (min_U_idx, U_norms[min_U_idx], 'U'),
            (min_V_idx, V_norms[min_V_idx], 'V'),
            (min_W_idx, W_norms[min_W_idx], 'W'),
        ]
        candidates.sort(key=lambda x: x[1])
        
        # Try removing the candidate with smallest norm
        idx, norm_val, mat_name = candidates[0]
        
        # Only remove if norm is small
        if norm_val < 0.1:
            # Remove column idx
            mask = np.ones(current_rank, dtype=bool)
            mask[idx] = False
            U_f = U_f[:, mask]
            V_f = V_f[:, mask]
            W_f = W_f[:, mask]
            current_rank -= 1
            
            # Verify
            ok, err = _verify_fast(U_f.astype(np.int16), V_f.astype(np.int16), W_f.astype(np.int16))
            if ok:
                best_rank = current_rank
                best_U = U_f.astype(np.int16)
                best_V = V_f.astype(np.int16)
                best_W = W_f.astype(np.int16)
            else:
                # Restore
                mask[idx] = True
                # Can't easily restore, so break
                break
        else:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_lbbfgs_optimize(U, V, W, n=4, m=4, p=8, time_budget=30.0):
    """
    Try L-BFGS-B optimization to find a lower-rank decomposition.
    This is a structural change: formulating as non-convex optimization.
    """
    start_time = time.time()
    
    # Start with the current decomposition
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    rank = U_f.shape[1]
    
    # Define the objective: Frobenius norm of residual
    def objective(params):
        if time.time() - start_time > time_budget:
            return 1e10
        
        # Unpack parameters
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute residual using random test matrices
        np.random.seed(42)
        A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
        B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
        C_expected = A @ B
        
        C_comp = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            Ar = A * U_flat[:, r].reshape(n, m)
            Br = B * V_flat[:, r].reshape(m, p)
            Mr = Ar @ Br
            Wr = W_flat[:, r].reshape(p, n).T
            C_comp += Mr * Wr
        
        residual = C_comp - C_expected
        return np.sum(residual**2)
    
    # Initial parameters
    x0 = np.concatenate([U_f.flatten(), V_f.flatten(), W_f.flatten()])
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 200})
        
        if result.success or result.fun < 1e-6:
            # Extract optimized matrices
            U_opt = result.x[:n*m*rank].reshape(n*m, rank)
            V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer
            U_opt = np.round(U_opt * 2) / 2
            V_opt = np.round(V_opt * 2) / 2
            W_opt = np.round(W_opt * 2) / 2
            
            # Verify
            ok, err = _verify_fast(U_opt.astype(np.int16) if np.allclose(U_opt, np.round(U_opt)) else U_opt,
                                   V_opt.astype(np.int16) if np.allclose(V_opt, np.round(V_opt)) else V_opt,
                                   W_opt.astype(np.int16) if np.allclose(W_opt, np.round(W_opt)) else W_opt)
            
            if ok:
                return U_opt, V_opt, W_opt, rank
    except Exception:
        pass
    
    return U, V, W, rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    U, V, W = _build_4x4x8()
    
    ok, err = _verify(U, V, W)
    if not ok:
        n, m, p = 4, 4, 8
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Try to reduce rank using optimization
    time_remaining = _time_left()
    if time_remaining > 20:
        try:
            U_opt, V_opt, W_opt, new_rank = _try_optimize(U, V, W, time_budget=min(40, time_remaining - 10))
            if new_rank < U.shape[1]:
                ok, err = _verify(U_opt, V_opt, W_opt)
                if ok:
                    U, V, W = U_opt, V_opt, W_opt
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
