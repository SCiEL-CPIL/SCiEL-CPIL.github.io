"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_N, _M, _P = 2, 3, 8
_NMP = _N * _M * _P  # 48

# Precompute target indices and mask
_TARGET_IDX = []
for _i in range(_N):
    for _j in range(_M):
        for _k in range(_P):
            _TARGET_IDX.append((_i * _M + _j, _j * _P + _k, _k * _N + _i))

_TARGET_MASK = np.zeros((_N*_M, _M*_P, _P*_N), dtype=bool)
for _a, _b, _c in _TARGET_IDX:
    _TARGET_MASK[_a, _b, _c] = True

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = 0.0
    for a, b, c in _TARGET_IDX:
        err = abs(T[a,b,c] - 1.0)
        if err > max_err:
            max_err = err
        if max_err > tol:
            return False
    
    non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
    return non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _compute_residual(U, V, W):
    """Compute residual = ||T - target||_F^2."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    residual = T.copy()
    for a, b, c in _TARGET_IDX:
        residual[a, b, c] -= 1.0
    return np.sum(residual**2)

def _precompute_als_indices():
    """Precompute all index arrays needed for ALS and polish."""
    n, m, p = DIMENSIONS
    
    U_a_idx = []
    U_b_idx = []
    U_c_idx = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                U_a_idx.append(a)
                U_b_idx.append(j * p + k)
                U_c_idx.append(k * n + i)
    U_a_idx = np.array(U_a_idx)
    U_b_idx = np.array(U_b_idx)
    U_c_idx = np.array(U_c_idx)
    
    V_a_idx = []
    V_c_idx = []
    V_row_count = 0
    V_rows_per_b = []
    for j in range(m):
        for k in range(p):
            b = j * p + k
            start = V_row_count
            for i in range(n):
                for kk in range(p):
                    V_a_idx.append(i * m + j)
                    V_c_idx.append(kk * n + i)
                    V_row_count += 1
            V_rows_per_b.append(V_row_count - start)
    V_a_idx = np.array(V_a_idx)
    V_c_idx = np.array(V_c_idx)
    
    W_a_idx = []
    W_b_idx = []
    W_row_count = 0
    W_rows_per_c = []
    for k in range(p):
        for i in range(n):
            c = k * n + i
            start = W_row_count
            for ii in range(n):
                for j in range(m):
                    W_a_idx.append(ii * m + j)
                    W_b_idx.append(j * p + k)
                    W_row_count += 1
            W_rows_per_c.append(W_row_count - start)
    W_a_idx = np.array(W_a_idx)
    W_b_idx = np.array(W_b_idx)
    
    return (U_a_idx, U_b_idx, U_c_idx,
            V_a_idx, V_c_idx, V_rows_per_b,
            W_a_idx, W_b_idx, W_rows_per_c)

_ALS_IDX = _precompute_als_indices()

def _als_decompose(rank, max_iters=100, deadline=None, seed=None, init=None):
    """
    Alternating Least Squares for tensor decomposition - vectorized.
    """
    n, m, p = DIMENSIONS
    
    if init is not None:
        U = np.array(init[0], dtype=float).copy()
        V = np.array(init[1], dtype=float).copy()
        W = np.array(init[2], dtype=float).copy()
    else:
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    best_residual = np.inf
    
    (U_a_idx, U_b_idx, U_c_idx,
     V_a_idx, V_c_idx, V_rows_per_b,
     W_a_idx, W_b_idx, W_rows_per_c) = _ALS_IDX
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            mask = U_a_idx == a_idx
            b_vals = U_b_idx[mask]
            c_vals = U_c_idx[mask]
            M = V[b_vals, :] * W[c_vals, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            start = sum(V_rows_per_b[:b_idx])
            end = start + V_rows_per_b[b_idx]
            a_vals = V_a_idx[start:end]
            c_vals = V_c_idx[start:end]
            M = U[a_vals, :] * W[c_vals, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                V[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            start = sum(W_rows_per_c[:c_idx])
            end = start + W_rows_per_c[c_idx]
            a_vals = W_a_idx[start:end]
            b_vals = W_b_idx[start:end]
            M = U[a_vals, :] * V[b_vals, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                W[c_idx, :] = x
            except:
                pass
                    
        T = np.einsum('ar,br,cr->abc', U, V, W)
            
        max_err = 0.0
        for (a, b, c) in _TARGET_IDX:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        
        non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        
        residual = _compute_residual(U, V, W)
        if residual < best_residual:
            best_residual = residual
    
    return None

def _polish(U, V, W, max_iters=50, tol=1e-14, deadline=None):
    """High-precision polish pass - vectorized."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    (U_a_idx, U_b_idx, U_c_idx,
     V_a_idx, V_c_idx, V_rows_per_b,
     W_a_idx, W_b_idx, W_rows_per_c) = _ALS_IDX
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        for a_idx in range(n*m):
            mask = U_a_idx == a_idx
            b_vals = U_b_idx[mask]
            c_vals = U_c_idx[mask]
            M = V_arr[b_vals, :] * W_arr[c_vals, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U_arr[a_idx, :] = x
            except:
                pass
                    
        for b_idx in range(m*p):
            start = sum(V_rows_per_b[:b_idx])
            end = start + V_rows_per_b[b_idx]
            a_vals = V_a_idx[start:end]
            c_vals = V_c_idx[start:end]
            M = U_arr[a_vals, :] * W_arr[c_vals, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                V_arr[b_idx, :] = x
            except:
                pass
                    
        for c_idx in range(p*n):
            start = sum(W_rows_per_c[:c_idx])
            end = start + W_rows_per_c[c_idx]
            a_vals = W_a_idx[start:end]
            b_vals = W_b_idx[start:end]
            M = U_arr[a_vals, :] * V_arr[b_vals, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                W_arr[c_idx, :] = x
            except:
                pass
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            
        max_err = 0.0
        for (a, b, c) in _TARGET_IDX:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
        
        if max_err < tol and non_target_err < tol:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
    return None

def _structured_rank_reduction(U44, V44, W44, target_rank, deadline, seed=0):
    """
    Try to reduce rank by exploiting block structure.
    """
    n, m, p = DIMENSIONS
    U44_arr = np.array(U44, dtype=float)
    V44_arr = np.array(V44, dtype=float)
    W44_arr = np.array(W44, dtype=float)
    
    rng = np.random.RandomState(seed)
    current_rank = 44
    U_cur = U44_arr.copy()
    V_cur = V44_arr.copy()
    W_cur = W44_arr.copy()
    
    max_merges = 44 - target_rank
    merges_done = 0
    
    for _ in range(max_merges * 100):
        if deadline and time.time() > deadline:
            break
        if current_rank == target_rank:
            break
        
        idx1 = rng.randint(0, current_rank)
        idx2 = rng.randint(0, current_rank)
        if idx1 == idx2:
            continue
        
        for attempt in range(50):
            alpha = rng.uniform(-3, 3)
            beta = rng.uniform(-3, 3)
            
            U_new = np.zeros((n*m, current_rank - 1))
            V_new = np.zeros((m*p, current_rank - 1))
            W_new = np.zeros((p*n, current_rank - 1))
            
            col = 0
            for i in range(current_rank):
                if i == idx1:
                    U_new[:, col] = alpha * U_cur[:, idx1] + beta * U_cur[:, idx2]
                    V_new[:, col] = alpha * V_cur[:, idx1] + beta * V_cur[:, idx2]
                    W_new[:, col] = alpha * W_cur[:, idx1] + beta * W_cur[:, idx2]
                    col += 1
                elif i == idx2:
                    continue
                else:
                    U_new[:, col] = U_cur[:, i]
                    V_new[:, col] = V_cur[:, i]
                    W_new[:, col] = W_cur[:, i]
                    col += 1
            
            if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), tol=1e-8):
                current_rank -= 1
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                merges_done += 1
                break
    
    if current_rank == target_rank:
        return U_cur.tolist(), V_cur.tolist(), W_cur.tolist()
    
    return None

def _strassen_block_decomposition(n, m, p, block_size=2):
    """
    Build a decomposition using Strassen-like structure on blocks of B.
    For each block of size block_size in the k-dimension:
    - Use Strassen for the 2x2 part (rows 0,1 of A and rows 0,1 of B block)
    - Use rank-1 for the remaining row of A (row 2)
    This gives 7 + 4 = 11 per block of size 2, so 44 total for p=8.
    
    But we can also try different block sizes or different Strassen variants.
    """
    n, m, p = 2, 3, 8
    num_blocks = p // block_size
    rank_per_block = 7 + 4  # Strassen(2x2) + 4 rank-1 for A[:,2]
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    for block in range(num_blocks):
        k0 = block * block_size
        k1 = block * block_size + 1
        
        # Strassen 7 products for A[0:2, 0:2] * B[0:2, k0:k1]
        # m1 = (a00+a11)(b0k0+b1k1)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b0k0
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b0k1-b1k1)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b1k0-b0k0)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b1k1
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b0k0+b0k1)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b1k0+b1k1)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 rank-1 terms: A[:,2] * B[2, k0:k1]
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == total_rank
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s kill
    
    # First, try the explicit rank-44 decomposition
    U44, V44, W44 = None, None, None
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception:
        pass
    
    # Try structured rank reduction from 44
    if best_rank == 44 and U44 is not None and time.time() < deadline:
        for target in [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28]:
            if time.time() > deadline - 10:
                break
            result = _structured_rank_reduction(U44, V44, W44, target, deadline, seed=target)
            if result:
                U, V, W = result
                if _verify(U, V, W, tol=1e-6):
                    rounded = _round_and_verify(U, V, W)
                    if rounded:
                        best_U, best_V, best_W = rounded
                    else:
                        best_U, best_V, best_W = U, V, W
                    best_rank = target
                    break
    
    # Try ALS from the rank-44 decomposition as initialization, targeting lower ranks
    if best_rank >= 40 and U44 is not None and time.time() < deadline:
        for rank in range(min(best_rank - 1, 43), 27, -1):
            if time.time() > deadline - 15:
                break
            U_init = np.array(U44, dtype=float)[:, :rank]
            V_init = np.array(V44, dtype=float)[:, :rank]
            W_init = np.array(W44, dtype=float)[:, :rank]
            
            result = _als_decompose(rank, max_iters=80, deadline=deadline, init=(U_init.tolist(), V_init.tolist(), W_init.tolist()))
            if result:
                U, V, W = result
                if _verify(U, V, W, tol=1e-6):
                    rounded = _round_and_verify(U, V, W)
                    if rounded:
                        best_U, best_V, best_W = rounded
                    else:
                        best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    # Try random ALS for lower ranks with multiple seeds
    if best_rank > 28 and time.time() < deadline:
        for rank in range(28, min(best_rank, 40)):
            if time.time() > deadline - 10:
                break
            for trial in range(12):
                if time.time() > deadline - 5:
                    break
                result = _als_decompose(rank, max_iters=80, deadline=deadline, seed=trial*1000+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        polished = _polish(U, V, W, max_iters=30, tol=1e-14, deadline=deadline)
                        if polished:
                            U, V, W = polished
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        break
            if best_rank < 44:
                break
    
    # Final polish if we have a non-schoolbook solution
    if best_rank < schoolbook_rank and time.time() < deadline - 3:
        polished = _polish(best_U, best_V, best_W, max_iters=20, tol=1e-14, deadline=deadline)
        if polished:
            best_U, best_V, best_W = polished
            rounded = _round_and_verify(best_U, best_V, best_W)
            if rounded:
                best_U, best_V, best_W = rounded
    
    return {"U": best_U, "V": best_V, "W": best_W}
