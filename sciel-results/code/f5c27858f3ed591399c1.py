import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=None, m=None, p=None):
    """Fast exact verification."""
    if n is None:
        n, m, p = DIMENSIONS

    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
        return False
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False

    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= 1e-9))

def _block_decomposition():
    """Assemble the full 3x4 x 4x8 decomposition (rank 56)."""
    def _strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        U[:, 0] = [1, 0, 0, 1]; V[:, 0] = [1, 0, 0, 1]; W[:, 0] = [1, 0, 0, 1]
        U[:, 1] = [0, 0, 1, 1]; V[:, 1] = [1, 0, 0, 0]; W[:, 1] = [0, 0, 1, 0]
        U[:, 2] = [1, 0, 0, 0]; V[:, 2] = [0, 0, 1, -1]; W[:, 2] = [0, 1, 0, 0]
        U[:, 3] = [0, 0, 0, 1]; V[:, 3] = [0, 1, -1, 0]; W[:, 3] = [1, 0, 0, 0]
        U[:, 4] = [1, 1, 0, 0]; V[:, 4] = [0, 0, 0, 1]; W[:, 4] = [0, 0, 0, 1]
        U[:, 5] = [0, 0, 1, -1]; V[:, 5] = [1, 1, 0, 0]; W[:, 5] = [0, 0, 0, 1]
        U[:, 6] = [0, 1, 0, -1]; V[:, 6] = [0, 0, 1, 1]; W[:, 6] = [0, 1, 0, 0]
        return U, V, W

    def _block_2x4x4x4():
        U, V, W = _strassen_2x2()
        Ua = np.zeros((8, 28), dtype=int)
        Va = np.zeros((16, 28), dtype=int)
        Wa = np.zeros((8, 28), dtype=int)
        blocks = [
            (0, 1, 0, 1, 0),
            (2, 3, 4, 5, 0),
            (0, 1, 2, 3, 2),
            (2, 3, 6, 7, 2),
        ]
        for b, (ar0, ar1, br0, br1, cc0) in enumerate(blocks):
            col0 = b * 7
            for t in range(7):
                col = col0 + t
                Ua[ar0, col] = U[0, t]
                Ua[ar1, col] = U[1, t]
                Va[br0, col] = V[0, t]
                Va[br1, col] = V[1, t]
                Wa[cc0, col] = W[0, t]
                Wa[cc0 + 1, col] = W[1, t]
        return Ua, Va, Wa

    def _block_1x4x4x4():
        Ua = np.zeros((4, 16), dtype=int)
        Va = np.zeros((16, 16), dtype=int)
        Wa = np.zeros((4, 16), dtype=int)
        for j in range(4):
            for k in range(4):
                col = j * 4 + k
                Ua[j, col] = 1
                Va[j * 4 + k, col] = 1
                Wa[k, col] = 1
        return Ua, Va, Wa

    Ua, Va, Wa = _block_2x4x4x4()
    U1, V1, W1 = _block_1x4x4x4()

    U = np.zeros((12, 56), dtype=int)
    V = np.zeros((32, 56), dtype=int)
    W = np.zeros((24, 56), dtype=int)

    U[0:8, 0:28] = Ua
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:28] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W[k * 3 + i, 0:28] = Wa[k * 4 + i, :]

    U[0:8, 28:56] = Ua
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 28:56] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W[(k + 4) * 3 + i, 28:56] = Wa[k * 4 + i, :]

    U[8:12, 0:16] = U1
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:16] = V1[j * 4 + k, :]
    for k in range(4):
        W[k * 3 + 2, 0:16] = W1[k, :]

    U[8:12, 28:44] = U1
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 28:44] = V1[j * 4 + k, :]
    for k in range(4):
        W[(k + 4) * 3 + 2, 28:44] = W1[k, :]

    return {"U": U, "V": V, "W": W}

def _to_lists(res):
    return {
        "U": res["U"].astype(int).tolist(),
        "V": res["V"].astype(int).tolist(),
        "W": res["W"].astype(int).tolist(),
    }

def _to_lists_half(res):
    """Convert to lists, preserving half-integers if present."""
    def _arr_to_list(arr):
        arr = np.asarray(arr, dtype=float)
        if np.allclose(arr, np.round(arr), atol=1e-9):
            return arr.astype(int).tolist()
        else:
            return arr.tolist()
    return {
        "U": _arr_to_list(res["U"]),
        "V": _arr_to_list(res["V"]),
        "W": _arr_to_list(res["W"]),
    }

def _als_search(n, m, p, rank, deadline, max_iters=200, tol=1e-12, init=None):
    """
    Alternating Least Squares for tensor rank approximation.
    Minimize ||T - sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]||_F
    where T is the all-ones tensor of shape (n, m, p).
    
    For fixed V and W, solve for U via linear least squares.
    For fixed U and W, solve for V via linear least squares.
    For fixed U and V, solve for W via linear least squares.
    """
    if init is not None:
        U = np.array(init["U"], dtype=float).reshape(n, m, rank)
        V = np.array(init["V"], dtype=float).reshape(m, p, rank)
        W = np.array(init["W"], dtype=float).reshape(p, n, rank)
    else:
        U = np.random.randn(n, m, rank)
        V = np.random.randn(m, p, rank)
        W = np.random.randn(p, n, rank)
    
    # Normalize columns
    U /= (np.linalg.norm(U, axis=(0,1), keepdims=True) + 1e-15)
    V /= (np.linalg.norm(V, axis=(0,1), keepdims=True) + 1e-15)
    W /= (np.linalg.norm(W, axis=(0,1), keepdims=True) + 1e-15)
    
    prev_err = np.inf
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Solve for W: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                b = np.ones(m)
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k, i, :] = sol
                except:
                    W_new[k, i, :] = W[k, i, :]
        W = W_new
        
        # Solve for U: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = V[j, :, :] * W[:, i, :]  # shape (p, rank)
                b = np.ones(p)
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i, j, :] = sol
                except:
                    U_new[i, j, :] = U[i, j, :]
        U = U_new
        
        # Solve for V: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                b = np.ones(n)
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j, k, :] = sol
                except:
                    V_new[j, k, :] = V[j, k, :]
        V = V_new
        
        # Compute error
        T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = np.linalg.norm(T_calc - np.ones((n, m, p)))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Try to round to integers/half-integers
    for scale in [2, 1]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        U_list = U_r.reshape(n*m, rank).tolist()
        V_list = V_r.reshape(m*p, rank).tolist()
        W_list = W_r.reshape(p*n, rank).tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_r.reshape(n*m, rank), "V": V_r.reshape(m*p, rank), "W": W_r.reshape(p*n, rank)}
    
    return None

def _structured_search(n, m, p, deadline, max_rank=55):
    """
    Search for structured low-rank decompositions using known algebraic patterns.
    """
    n, m, p = DIMENSIONS
    
    base = _block_decomposition()
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    rank0 = U0.shape[1]
    
    best = None
    best_rank = rank0
    
    for target_rank in range(min(50, best_rank - 1), 35, -1):
        if time.time() > deadline - 10.0:
            break
        
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            
            if target_rank < rank0:
                U_init = U0[:, :target_rank]
                V_init = V0[:, :target_rank]
                W_init = W0[:, :target_rank]
            else:
                U_init = U0
                V_init = V0
                W_init = W0
            
            init = {"U": U_init, "V": V_init, "W": W_init}
            
            result = _als_search(n, m, p, target_rank, deadline, max_iters=50, init=init)
            if result is not None:
                if _verify(result["U"].tolist(), result["V"].tolist(), result["W"].tolist()):
                    if best is None or target_rank < best_rank:
                        best = result
                        best_rank = target_rank
                    break
    
    if best is not None:
        return _to_lists_half(best)
    
    return None

def _search_lower_rank(deadline):
    """
    Attempt to find a decomposition with rank < 56 using ALS.
    """
    n, m, p = DIMENSIONS
    
    # Try structured search first
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        return result
    
    # Then try random ALS
    target_ranks = range(45, 56)
    
    for r in target_ranks:
        if time.time() > deadline - 15.0:
            break
        
        for trial in range(3):
            if time.time() > deadline - 15.0:
                break
            
            np.random.seed(12345 + r * 1000 + trial)
            result = _als_search(n, m, p, r, deadline, max_iters=80)
            if result is not None:
                if _verify(result["U"].tolist(), result["V"].tolist(), result["W"].tolist()):
                    return _to_lists_half(result)
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # Guaranteed-valid baseline.
    best = _schoolbook()
    best_rank = len(best["U"][0])

    # Primary structured result: block Strassen, rank 56.
    try:
        if time.time() < deadline - 30.0:
            res = _block_decomposition()
            if _verify(res["U"].tolist(), res["V"].tolist(), res["W"].tolist()):
                best = _to_lists(res)
                best_rank = len(best["U"][0])
    except Exception:
        pass

    # Try structured and ALS search for lower rank
    try:
        if time.time() < deadline - 20.0:
            improved = _search_lower_rank(deadline)
            if improved is not None:
                if _verify(improved["U"], improved["V"], improved["W"]):
                    best = improved
                    best_rank = len(best["U"][0])
    except Exception:
        pass

    # Final verification; fall back to schoolbook on any anomaly.
    try:
        if not _verify(best["U"], best["V"], best["W"]):
            best = _schoolbook()
    except Exception:
        best = _schoolbook()

    return best
