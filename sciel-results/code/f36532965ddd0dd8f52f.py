"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_EARLY_DEADLINE = 145  # seconds from start, leaves 5s safety margin

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
    
    non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
    
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
    
    # Try quarters
    U_q = np.round(U_arr * 4) / 4
    V_q = np.round(V_arr * 4) / 4
    W_q = np.round(W_arr * 4) / 4
    if _verify(U_q.tolist(), V_q.tolist(), W_q.tolist(), tol=tol):
        return U_q.tolist(), V_q.tolist(), W_q.tolist()
    
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    
    Strategy: Split B (3x8) into 4 blocks of 3x2: B = [B_0, B_1, B_2, B_3]
    Each C_t = A * B_t is a 2x3 * 3x2 = 2x2 multiplication.
    
    For each 2x3*3x2 sub-problem, use rank-11 scheme:
    - Strassen rank-7 for the 2x2 output (treating A as having 3 columns)
    - Plus 4 rank-1 terms to handle the extra dimension
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        # Block handles columns 2*block and 2*block+1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for the 2x2 * 2x2 part (A cols 0,1 times B rows 0,1)
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1   # a00
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1   # a10
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        W[2*k0+1, t] += 1  # C10
        W[2*k1+1, t] += -1  # C11
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1   # a00
        V[k1, t] += 1  # b01
        V[8+k1, t] += -1  # b11
        W[2*k1, t] += 1  # C01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1   # a11
        V[8+k0, t] += 1  # b10
        V[k0, t] += -1  # b00
        W[2*k0, t] += 1  # C00
        W[2*k0+1, t] += 1  # C10
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1   # a00
        U[1, t] += 1   # a01
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += -1  # C00
        W[2*k1, t] += 1  # C01
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1   # a10
        U[0, t] += -1  # a00
        V[k0, t] += 1  # b00
        V[k1, t] += 1  # b01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1   # a01
        U[4, t] += -1  # a11
        V[8+k0, t] += 1  # b10
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        col_offset += 1
        
        # Now the 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1   # A[ii, 2]
                V[16+k_val, t] = 1  # B[2, k_val]
                W[k_val*2+ii, t] = 1  # C[ii, k_val]
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _precompute_indices():
    """Precompute index mappings for faster ALS iterations."""
    n, m, p = DIMENSIONS
    # For U update: for each (i,j), list of (k, b, c)
    U_indices = []
    for i in range(n):
        for j in range(m):
            entries = []
            for k in range(p):
                b = j * p + k
                c = k * n + i
                entries.append((k, b, c))
            U_indices.append(entries)
    
    # For V update: for each (j,k), list of (i, kk, a, c)
    V_indices = []
    for j in range(m):
        for k in range(p):
            entries = []
            for i in range(n):
                for kk in range(p):
                    a = i * m + j
                    c = kk * n + i
                    entries.append((i, kk, a, c))
            V_indices.append(entries)
    
    # For W update: for each (k,i), list of (ii, j, a, b)
    W_indices = []
    for k in range(p):
        for i in range(n):
            entries = []
            for ii in range(n):
                for j in range(m):
                    a = ii * m + j
                    b = j * p + k
                    entries.append((ii, j, a, b))
            W_indices.append(entries)
    
    # Target indices for verification
    target_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_idx.append((i*m+j, j*p+k, k*n+i))
    
    return U_indices, V_indices, W_indices, target_idx

_U_indices, _V_indices, _W_indices, _target_idx = _precompute_indices()

# Precompute target mask globally
_n, _m, _p = DIMENSIONS
_target_mask = np.zeros((_n*_m, _m*_p, _p*_n), dtype=bool)
for (a, b, c) in _target_idx:
    _target_mask[a, b, c] = True

def _als_decompose(rank, max_iters=100, deadline=None, seed=None):
    """
    Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            i = a_idx // m
            j = a_idx % m
            entries = _U_indices[a_idx]
            M = np.zeros((p, rank))
            for row, (k, b, c) in enumerate(entries):
                M[row, :] = V[b, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            j = b_idx // p
            k = b_idx % p
            entries = _V_indices[b_idx]
            M = np.zeros((len(entries), rank))
            for row, (i, kk, a, c) in enumerate(entries):
                M[row, :] = U[a, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            k = c_idx // n
            i = c_idx % n
            entries = _W_indices[c_idx]
            M = np.zeros((len(entries), rank))
            for row, (ii, j, a, b) in enumerate(entries):
                M[row, :] = U[a, :] * V[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W[c_idx, :] = x
            except:
                pass
                    
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for (a, b, c) in _target_idx:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        
        non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, max_iters=50, tol=1e-12, deadline=None):
    """High-precision polish pass using tighter tolerance and more iterations."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            entries = _U_indices[a_idx]
            M = np.zeros((p, rank))
            for row, (k, b, c) in enumerate(entries):
                M[row, :] = V_arr[b, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U_arr[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            entries = _V_indices[b_idx]
            M = np.zeros((len(entries), rank))
            for row, (i, kk, a, c) in enumerate(entries):
                M[row, :] = U_arr[a, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V_arr[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            entries = _W_indices[c_idx]
            M = np.zeros((len(entries), rank))
            for row, (ii, j, a, b) in enumerate(entries):
                M[row, :] = U_arr[a, :] * V_arr[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W_arr[c_idx, :] = x
            except:
                pass
        
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
            
        max_err = 0.0
        for (a, b, c) in _target_idx:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
        
        if max_err < tol and non_target_err < tol:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
    return None

def _als_from_init(U_init, V_init, W_init, target_rank, max_iters=100, deadline=None):
    """ALS starting from a given initialization, targeting a specific rank."""
    n, m, p = DIMENSIONS
    cur_rank = len(U_init[0])
    
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    # If target_rank < cur_rank, randomly select target_rank columns
    if target_rank < cur_rank:
        idx = np.random.choice(cur_rank, target_rank, replace=False)
        U = U[:, idx]
        V = V[:, idx]
        W = W[:, idx]
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            entries = _U_indices[a_idx]
            M = np.zeros((p, target_rank))
            for row, (k, b, c) in enumerate(entries):
                M[row, :] = V[b, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            entries = _V_indices[b_idx]
            M = np.zeros((len(entries), target_rank))
            for row, (i, kk, a, c) in enumerate(entries):
                M[row, :] = U[a, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            entries = _W_indices[c_idx]
            M = np.zeros((len(entries), target_rank))
            for row, (ii, j, a, b) in enumerate(entries):
                M[row, :] = U[a, :] * V[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W[c_idx, :] = x
            except:
                pass
                    
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for (a, b, c) in _target_idx:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
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
    deadline = start_time + _EARLY_DEADLINE
    
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
    except Exception as e:
        pass
    
    # Try to find better rank using warm-start from rank-44
    if best_rank == 44 and U44 is not None:
        for rank in range(30, best_rank):
            if time.time() > deadline:
                break
            
            found = False
            for trial in range(15):
                if time.time() > deadline:
                    break
                np.random.seed(trial * 7919 + rank * 131)
                result = _als_from_init(U44, V44, W44, rank, max_iters=80, deadline=deadline)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        polished = _polish(U, V, W, max_iters=50, tol=1e-12, deadline=deadline)
                        if polished:
                            U, V, W = polished
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            found = True
                            break
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                            found = True
                            break
            
            if found:
                break
    
    # Also try random ALS for remaining ranks
    if best_rank == 44:
        for rank in range(30, best_rank):
            if time.time() > deadline:
                break
            
            for trial in range(8):
                if time.time() > deadline:
                    break
                
                result = _als_decompose(rank, max_iters=60, deadline=deadline, seed=trial*1000+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        polished = _polish(U, V, W, max_iters=40, tol=1e-12, deadline=deadline)
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
