"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
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
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
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
        
        # Rank-4 terms for A_right * B_bottom
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

def _als_decompose(rank, max_iters=100, deadline=None, init=None, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition with polish.
    """
    n, m, p = DIMENSIONS
    
    if init:
        U = init[0].copy()
        V = init[1].copy()
        W = init[2].copy()
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    # Precompute target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
        
        # Check convergence every few iterations
        if it % 5 == 0 or it == max_iters - 1:
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            max_err = 0.0
            for (a, b, c) in target_indices:
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
            
            target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
            for (a, b, c) in target_indices:
                target_mask[a,b,c] = True
            non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
            
            if max_err < tol and non_target_err < tol:
                return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, max_iters=50, deadline=None, tol=1e-12):
    """High-precision polish pass."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
        
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        max_err = 0.0
        for (a, b, c) in target_indices:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        
        target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for (a, b, c) in target_indices:
            target_mask[a,b,c] = True
        non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
        
        if max_err < tol and non_target_err < tol:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

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
    deadline = start_time + 140
    
    # First, try the explicit rank-44 decomposition
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
    
    # Now try to find better with ALS, starting from rank 43 down to 35
    if best_rank == 44:
        for rank in range(43, 34, -1):
            if time.time() > deadline:
                break
            
            # More trials for ranks closer to 44
            trials = 8 if rank >= 41 else 5
            
            for trial in range(trials):
                if time.time() > deadline:
                    break
                
                result = _als_decompose(rank, max_iters=80, deadline=deadline)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        # Polish with tighter tolerance
                        polished = _polish(U, V, W, max_iters=30, deadline=deadline, tol=1e-12)
                        if polished:
                            U, V, W = polished
                        
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            break
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                            break
            
            if best_rank < 44:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
