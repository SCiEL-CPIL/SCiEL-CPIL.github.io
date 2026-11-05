"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_altsolve(n, m, p, rank, max_iter=500, seed=None, tol=1e-10):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        Ua_snapped = np.vectorize(_snap_to_half)(best_U)
        Va_snapped = np.vectorize(_snap_to_half)(best_V)
        Wa_snapped = np.vectorize(_snap_to_half)(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_gradient(n, m, p, rank, max_iter=3000, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    lr = 0.05
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        error = np.max(np.abs(err))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if error < 1e-10:
            break
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        
        U3 -= lr * grad_U
        V3 -= lr * grad_V
        W3 -= lr * grad_W
        
        if it % 100 == 0:
            lr *= 0.8
    
    Ua = best_U.reshape(n*m, rank)
    Va = best_V.reshape(m*p, rank)
    Wa = best_W.reshape(p*n, rank)
    
    Ua_snapped = np.vectorize(_snap_to_half)(Ua)
    Va_snapped = np.vectorize(_snap_to_half)(Va)
    Wa_snapped = np.vectorize(_snap_to_half)(Wa)
    
    U_out = Ua_snapped.tolist()
    V_out = Va_snapped.tolist()
    W_out = Wa_snapped.tolist()
    
    if _verify_fast(U_out, V_out, W_out, n, m, p):
        return U_out, V_out, W_out
    
    U_out_raw = Ua.tolist()
    V_out_raw = Va.tolist()
    W_out_raw = Wa.tolist()
    
    if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
        return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_structure(n, m, p, rank, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        i = rng.randint(0, n-1)
        j = rng.randint(0, m-1)
        k = rng.randint(0, p-1)
        
        U[i*m+j, r] = rng.uniform(0.5, 1.5)
        V[j*p+k, r] = rng.uniform(0.5, 1.5)
        W[k*n+i, r] = rng.uniform(0.5, 1.5)
        
        if rng.random() < 0.3:
            i2 = rng.randint(0, n-1)
            j2 = rng.randint(0, m-1)
            k2 = rng.randint(0, p-1)
            U[i2*m+j2, r] += rng.uniform(-0.5, 0.5)
            V[j2*p+k2, r] += rng.uniform(-0.5, 0.5)
            W[k2*n+i2, r] += rng.uniform(-0.5, 0.5)
    
    return U, V, W

def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = Va[j*p + k] * Wa[k*n + i]
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = Ua[i*m + j] * Wa[k*n + i]
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = Ua[i*m + j] * Va[j*p + k]
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _try_kronecker_structure(n, m, p, rank, seed=None):
    """
    Try a Kronecker-product-based structured initialization.
    Decompose indices into binary-like sub-indices and use structured patterns.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Try to use a block structure
    # For 3x5x6, we can think of 3 = 1*3, 5 = 1*5, 6 = 2*3
    # Or use binary representations
    
    for r in range(rank):
        # Assign a random "type" to this rank component
        comp_type = rng.randint(0, 3)
        
        if comp_type == 0:
            # Type 0: affects a single (i,j,k)
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            U[i*m+j, r] = rng.uniform(0.5, 1.5)
            V[j*p+k, r] = rng.uniform(0.5, 1.5)
            W[k*n+i, r] = rng.uniform(0.5, 1.5)
        elif comp_type == 1:
            # Type 1: affects a row of i (all j, k)
            i = rng.randint(0, n-1)
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] += rng.uniform(0.1, 0.5)
                    V[j*p+k, r] += rng.uniform(0.1, 0.5)
                    W[k*n+i, r] += rng.uniform(0.1, 0.5)
        elif comp_type == 2:
            # Type 2: affects a column of j
            j = rng.randint(0, m-1)
            for i in range(n):
                for k in range(p):
                    U[i*m+j, r] += rng.uniform(0.1, 0.5)
                    V[j*p+k, r] += rng.uniform(0.1, 0.5)
                    W[k*n+i, r] += rng.uniform(0.1, 0.5)
        else:
            # Type 3: affects a slice of k
            k = rng.randint(0, p-1)
            for i in range(n):
                for j in range(m):
                    U[i*m+j, r] += rng.uniform(0.1, 0.5)
                    V[j*p+k, r] += rng.uniform(0.1, 0.5)
                    W[k*n+i, r] += rng.uniform(0.1, 0.5)
    
    return U, V, W

def _try_low_rank_exact(n, m, p, rank, max_iter=200, seed=None):
    """
    Try to find an exact low-rank decomposition using a combination of
    structured initialization and alternating linear solve.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Try multiple initializations
    for init_type in range(4):
        if _remaining() < 5.0:
            break
        
        if init_type == 0:
            U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
            V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
            W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        elif init_type == 1:
            U, V, W = _try_rank_structure(n, m, p, rank, seed=seed+init_type)
        elif init_type == 2:
            U, V, W = _try_kronecker_structure(n, m, p, rank, seed=seed+init_type)
        else:
            # Sparse random initialization
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                for _ in range(rng.randint(1, 3)):
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j, r] = rng.choice([-1, 1])
                    V[j*p+k, r] = rng.choice([-1, 1])
                    W[k*n+i, r] = rng.choice([-1, 1])
        
        # Run alternating solve
        for it in range(max_iter):
            if _remaining() < 3.0:
                break
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p + k] * W[k*n + i]
                    try:
                        U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m + j] * W[k*n + i]
                    try:
                        V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m + j] * V[j*p + k]
                    try:
                        W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-10:
                U_snapped = np.vectorize(_snap_to_half)(U).tolist()
                V_snapped = np.vectorize(_snap_to_half)(V).tolist()
                W_snapped = np.vectorize(_snap_to_half)(W).tolist()
                if _verify_fast(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                else:
                    return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 multiplication with rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[2, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _try_strassen_init(n, m, p, rank, seed=None):
    """
    Try to initialize using Strassen blocks where possible.
    This provides a structured starting point for the alternating solver.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # This is a placeholder for a more complex block construction
    # For now, we just use random init but with a bias towards non-zero values
    # that might help the solver converge faster
    
    for r in range(rank):
        i = rng.randint(0, n-1)
        j = rng.randint(0, m-1)
        k = rng.randint(0, p-1)
        U[i*m+j, r] = rng.uniform(-1, 1)
        V[j*p+k, r] = rng.uniform(-1, 1)
        W[k*n+i, r] = rng.uniform(-1, 1)
        
        # Add some structure
        if rng.random() < 0.5:
            i2 = (i + 1) % n
            j2 = (j + 1) % m
            k2 = (k + 1) % p
            U[i2*m+j2, r] += rng.uniform(-0.5, 0.5)
            V[j2*p+k2, r] += rng.uniform(-0.5, 0.5)
            W[k2*n+i2, r] += rng.uniform(-0.5, 0.5)
            
    return U, V, W

def _try_rank_grid_search(n, m, p, rank, max_iter=5000, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Initialize with small random integers or half-integers
    values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    
    def rand_val():
        return rng.choice(values)

    U = np.array([[rand_val() for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rand_val() for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rand_val() for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    def get_error():
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))
    
    current_error = get_error()
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        tensor_id = rng.randint(0, 2)
        if tensor_id == 0:
            idx = rng.randint(0, n*m-1)
            r = rng.randint(0, rank-1)
            old_val = U[idx, r]
            delta = rng.choice([-0.5, 0.5])
            new_val = old_val + delta
            if abs(new_val) > 5.0:
                continue
            U[idx, r] = new_val
            U3 = U.reshape(n, m, rank)
        elif tensor_id == 1:
            idx = rng.randint(0, m*p-1)
            r = rng.randint(0, rank-1)
            old_val = V[idx, r]
            delta = rng.choice([-0.5, 0.5])
            new_val = old_val + delta
            if abs(new_val) > 5.0:
                continue
            V[idx, r] = new_val
            V3 = V.reshape(m, p, rank)
        else:
            idx = rng.randint(0, p*n-1)
            r = rng.randint(0, rank-1)
            old_val = W[idx, r]
            delta = rng.choice([-0.5, 0.5])
            new_val = old_val + delta
            if abs(new_val) > 5.0:
                continue
            W[idx, r] = new_val
            W3 = W.reshape(p, n, rank)
            
        new_error = get_error()
        
        if new_error < current_error:
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error < 1e-9:
                    break
        else:
            if tensor_id == 0:
                U[idx, r] = old_val
                U3 = U.reshape(n, m, rank)
            elif tensor_id == 1:
                V[idx, r] = old_val
                V3 = V.reshape(m, p, rank)
            else:
                W[idx, r] = old_val
                W3 = W.reshape(p, n, rank)
                
    if best_error < 1e-9:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    min_rank = max(n*m, m*p, n*p)
    
    # Try grid search for a range of ranks
    for rank in range(min_rank, min_rank + 15):
        if _remaining() < 20.0:
            break
        
        for seed in range(5):
            if _remaining() < 10.0:
                break
                
            result = _try_rank_grid_search(n, m, p, rank, max_iter=2000, seed=rank*1000+seed)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return best
    
    # Fallback to old methods if grid search fails
    if best_rank == n * m * p:
        for rank in range(min_rank, min_rank + 5):
            if _remaining() < 10.0:
                break
            for attempt in range(3):
                result = _try_low_rank_exact(n, m, p, rank, max_iter=100, seed=rank*100+attempt)
                if result is not None:
                    U, V, W = result
                    if _verify_fast(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        break
            if best_rank < n * m * p:
                break

    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
        
    return best
