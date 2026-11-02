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
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    target_a = []
    target_b = []
    target_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_a.append(i * m + j)
                target_b.append(j * p + k)
                target_c.append(k * n + i)
    
    target_vals = T[np.array(target_a), np.array(target_b), np.array(target_c)]
    max_err = np.max(np.abs(target_vals - 1.0))
    
    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for a, b, c in zip(target_a, target_b, target_c):
        target_mask[a, b, c] = True
    non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
    
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
    Split B (3x8) into 4 blocks of 3x2. Each block: Strassen-7 + 4 rank-1 terms.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for 2x2 * 2x2 (A rows 0,1 cols 0,1 * B rows 0,1 cols k0,k1)
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 rank-1 terms: A[i,2] * B[2,k] -> C[i,k]
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

def _precompute_als_data():
    """Precompute all data needed for vectorized ALS."""
    n, m, p = DIMENSIONS
    
    # Target indices
    target_a = []
    target_b = []
    target_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_a.append(i * m + j)
                target_b.append(j * p + k)
                target_c.append(k * n + i)
    
    # For U[a] update: need V[b]*W[c] for all (b,c) where target exists with this a
    # U[a] where a = i*m+j: targets are (a, j*p+k, k*n+i) for k in range(p)
    U_M_list = []  # list of (p, rank) matrices to be built
    U_b_list = []  # b indices
    U_c_list = []  # c indices
    for i in range(n):
        for j in range(m):
            a = i * m + j
            bs = []
            cs = []
            for k in range(p):
                bs.append(j * p + k)
                cs.append(k * n + i)
            U_M_list.append((bs, cs))
    
    # For V[b] update: b = j*p+k: targets are (i*m+j, b, k*n+i) for i in range(n)
    # But also need to consider all i, so for fixed b=(j,k), targets have a=i*m+j, c=k*n+i
    V_M_list = []
    for j in range(m):
        for k in range(p):
            b = j * p + k
            as_ = []
            cs = []
            for i in range(n):
                as_.append(i * m + j)
                cs.append(k * n + i)
            V_M_list.append((as_, cs))
    
    # For W[c] update: c = k*n+i: targets are (ii*m+j, j*p+k, c) for ii in range(n), j in range(m)
    W_M_list = []
    for k in range(p):
        for i in range(n):
            c = k * n + i
            as_ = []
            bs = []
            for ii in range(n):
                for j in range(m):
                    as_.append(ii * m + j)
                    bs.append(j * p + k)
            W_M_list.append((as_, bs))
    
    target_a = np.array(target_a)
    target_b = np.array(target_b)
    target_c = np.array(target_c)
    
    return U_M_list, V_M_list, W_M_list, target_a, target_b, target_c

_U_M_list, _V_M_list, _W_M_list, _target_a, _target_b, _target_c = _precompute_als_data()

def _als_decompose(rank, max_iters=80, deadline=None, seed=None):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Precompute target mask
    target_mask = np.zeros((nm, mp, pn), dtype=bool)
    for a, b, c in zip(_target_a, _target_b, _target_c):
        target_mask[a, b, c] = True
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U: for each a, solve M @ x = ones_p
        for idx, (bs, cs) in enumerate(_U_M_list):
            M = np.zeros((p, rank))
            for row, (b, c) in enumerate(zip(bs, cs)):
                M[row, :] = V[b, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U[idx, :] = x
            except:
                pass
                    
        # Update V: for each b, solve M @ x = ones_np
        for idx, (as_, cs) in enumerate(_V_M_list):
            M = np.zeros((n, rank))
            for row, (a, c) in enumerate(zip(as_, cs)):
                M[row, :] = U[a, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V[idx, :] = x
            except:
                pass
                    
        # Update W: for each c, solve M @ x = ones_nm
        for idx, (as_, bs) in enumerate(_W_M_list):
            M = np.zeros((n*m, rank))
            for row, (a, b) in enumerate(zip(as_, bs)):
                M[row, :] = U[a, :] * V[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W[idx, :] = x
            except:
                pass
                    
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target_vals = T[_target_a, _target_b, _target_c]
        max_err = np.max(np.abs(target_vals - 1.0))
        non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, max_iters=30, tol=1e-12, deadline=None):
    """High-precision polish pass."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for a, b, c in zip(_target_a, _target_b, _target_c):
        target_mask[a, b, c] = True
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        for idx, (bs, cs) in enumerate(_U_M_list):
            M = np.zeros((p, rank))
            for row, (b, c) in enumerate(zip(bs, cs)):
                M[row, :] = V_arr[b, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U_arr[idx, :] = x
            except:
                pass
                    
        for idx, (as_, cs) in enumerate(_V_M_list):
            M = np.zeros((n, rank))
            for row, (a, c) in enumerate(zip(as_, cs)):
                M[row, :] = U_arr[a, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V_arr[idx, :] = x
            except:
                pass
                    
        for idx, (as_, bs) in enumerate(_W_M_list):
            M = np.zeros((n*m, rank))
            for row, (a, b) in enumerate(zip(as_, bs)):
                M[row, :] = U_arr[a, :] * V_arr[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W_arr[idx, :] = x
            except:
                pass
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        target_vals = T[_target_a, _target_b, _target_c]
        max_err = np.max(np.abs(target_vals - 1.0))
        non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
        
        if max_err < tol and non_target_err < tol:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
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
    
    # Now try to find better with ALS
    if best_rank > 30:
        for rank in range(30, best_rank):
            if time.time() > deadline:
                break
            
            found = False
            for trial in range(8):
                if time.time() > deadline:
                    break
                
                result = _als_decompose(rank, max_iters=60, deadline=deadline, seed=trial*1000+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        polished = _polish(U, V, W, max_iters=20, tol=1e-12, deadline=deadline)
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
