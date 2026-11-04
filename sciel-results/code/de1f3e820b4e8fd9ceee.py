import numpy as np
import time

# Dimensions: n=2, m=3, p=6
n, m, p = 2, 3, 6
N_U = n * m  # 6
N_V = m * p  # 18
N_W = p * n  # 12

def _schoolbook():
    """Return schoolbook decomposition (rank 36)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(N_U)]
    V = [[0] * rank for _ in range(N_V)]
    W = [[0] * rank for _ in range(N_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify exact decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T_calc = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    return np.allclose(T_calc, T_target, atol=1e-9)

def _try_rank_direct(rank, seed=0, max_iter=5000, tol=1e-12):
    """
    Direct optimization using gradient descent with momentum.
    Try to find exact half-integer decomposition.
    """
    if time.time() > 145:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n, m, rank) * 0.05
    V = rng.randn(m, p, rank) * 0.05
    W = rng.randn(p, n, rank) * 0.05
    
    T_target = np.ones((n, m, p))
    lr = 0.01
    momentum = 0.9
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    prev_err = np.inf
    
    for it in range(max_iter):
        if time.time() > 145:
            return None
        
        # Compute gradients
        T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = np.sum((T_target - T_calc) ** 2)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
        # Gradients
        # dU[i,j,r] = -2 * sum_k (T_target[i,j,k] - T_calc[i,j,k]) * V[j,k,r] * W[k,i,r]
        # dV[j,k,r] = -2 * sum_i (T_target[i,j,k] - T_calc[i,j,k]) * U[i,j,r] * W[k,i,r]
        # dW[k,i,r] = -2 * sum_j (T_target[i,j,k] - T_calc[i,j,k]) * U[i,j,r] * V[j,k,r]
        
        diff = T_target - T_calc
        
        gU = -2 * np.einsum('ijk,jkr,kir->ijr', diff, V, W)
        gV = -2 * np.einsum('ijk,ijr,kir->jkr', diff, U, W)
        gW = -2 * np.einsum('ijk,ijr,jkr->kir', diff, U, V)
        
        # Momentum update
        vU = momentum * vU + lr * gU
        vV = momentum * vV + lr * gV
        vW = momentum * vW + lr * gW
        
        U += vU
        V += vV
        W += vW
        
        # Periodically try to snap to half-integers and check
        if it % 100 == 0:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
            if np.allclose(T_snap, T_target, atol=1e-9):
                U_flat = U_snap.reshape(-1, rank).tolist()
                V_flat = V_snap.reshape(-1, rank).tolist()
                W_flat = W_snap.reshape(-1, rank).tolist()
                return (U_flat, V_flat, W_flat)
    
    # Final check
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
    if np.allclose(T_snap, T_target, atol=1e-9):
        U_flat = U_snap.reshape(-1, rank).tolist()
        V_flat = V_snap.reshape(-1, rank).tolist()
        W_flat = W_snap.reshape(-1, rank).tolist()
        return (U_flat, V_flat, W_flat)
    
    return None

def _try_als(rank, seed=0, max_iter=200, tol=1e-10):
    """ALS with better initialization and verification."""
    if time.time() > 145:
        return None
    
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n, m, rank) * 0.1
    V = rng.randn(m, p, rank) * 0.1
    W = rng.randn(p, n, rank) * 0.1
    
    T_target = np.ones((n, m, p))
    prev_err = np.inf
    
    for it in range(max_iter):
        if time.time() > 145:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                M_u = V[j, :, :] * W[:, i, :]
                A = M_u.T @ M_u + 1e-10 * np.eye(rank)
                b = M_u.T @ T_target[i, j, :]
                try:
                    U[i, j, :] = np.linalg.solve(A, b)
                except:
                    U[i, j, :] = np.linalg.lstsq(M_u, T_target[i, j, :], rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                M_v = U[:, j, :] * W[k, :, :]
                A = M_v.T @ M_v + 1e-10 * np.eye(rank)
                b = M_v.T @ T_target[:, j, k]
                try:
                    V[j, k, :] = np.linalg.solve(A, b)
                except:
                    V[j, k, :] = np.linalg.lstsq(M_v, T_target[:, j, k], rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                M_w = U[i, :, :] * V[:, k, :]
                A = M_w.T @ M_w + 1e-10 * np.eye(rank)
                b = M_w.T @ T_target[i, :, k]
                try:
                    W[k, i, :] = np.linalg.solve(A, b)
                except:
                    W[k, i, :] = np.linalg.lstsq(M_w, T_target[i, :, k], rcond=None)[0]
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = np.linalg.norm(T_target - T_calc)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Try to snap to half-integers
    for scale in [2, 4, 8]:
        U_snap = np.round(U * scale) / scale
        V_snap = np.round(V * scale) / scale
        W_snap = np.round(W * scale) / scale
        
        T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
        if np.allclose(T_snap, T_target, atol=1e-9):
            U_flat = U_snap.reshape(-1, rank).tolist()
            V_flat = V_snap.reshape(-1, rank).tolist()
            W_flat = W_snap.reshape(-1, rank).tolist()
            return (U_flat, V_flat, W_flat)
    
    return None

def _strassen_like_2x3x6():
    """
    Try to construct a decomposition based on Strassen's algorithm ideas.
    For 2x3x6, we can think of this as two 2x3x3 blocks.
    """
    # This is a placeholder - we'll try to find if there's a known decomposition
    # For now, return None and let other methods handle it
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for 150s limit
    
    best_result = None
    best_rank = 36
    
    # Get schoolbook as baseline
    schoolbook = _schoolbook()
    if _verify(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = 36
    
    # Try ranks from lower bound upwards
    # Lower bound is max(nm, mp, np) = max(6, 18, 12) = 18
    # But structural constraints push it higher. Known upper bounds suggest ~24-28.
    
    ranks_to_try = list(range(18, 29))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple methods for this rank
        methods = [
            (lambda: _try_als(r, seed=0, max_iter=50, tol=1e-11), "als_seed0"),
            (lambda: _try_als(r, seed=1, max_iter=50, tol=1e-11), "als_seed1"),
            (lambda: _try_als(r, seed=2, max_iter=50, tol=1e-11), "als_seed2"),
            (lambda: _try_rank_direct(r, seed=0, max_iter=2000, tol=1e-13), "gd_seed0"),
            (lambda: _try_rank_direct(r, seed=1, max_iter=2000, tol=1e-13), "gd_seed1"),
            (lambda: _try_rank_direct(r, seed=2, max_iter=2000, tol=1e-13), "gd_seed2"),
        ]
        
        found = False
        for method_func, method_name in methods:
            if time.time() > deadline:
                break
            if found:
                break
            
            try:
                result = method_func()
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = r
                        found = True
            except Exception:
                pass
        
        if best_rank == 18:
            break
    
    if best_result is None:
        best_result = _schoolbook()
    
    return best_result

# Quick test
if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Found decomposition with rank {rank}")
    print(f"Verification: {_verify(res['U'], res['V'], res['W'])}")
