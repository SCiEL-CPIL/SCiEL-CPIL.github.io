import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
HARD_LIMIT = 145.0  # safety margin for 150s wall clock

def _verify_decomposition(U, V, W, n, m, p):
    """Exact integer verification using einsum."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    U_r = U_arr.reshape(n, m, r)
    V_r = V_arr.reshape(m, p, r)
    W_r = W_arr.reshape(p, n, r)
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    except Exception:
        return False
    return bool(np.all(reconstructed == 1))

def _verify_float(U, V, W, n, m, p, atol=1e-6):
    """Fast float verification."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_r = U_arr.reshape(n, m, r)
    V_r = V_arr.reshape(m, p, r)
    W_r = W_arr.reshape(p, n, r)
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    except Exception:
        return False
    return bool(np.allclose(reconstructed, 1.0, atol=atol))

def _schoolbook():
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

def _strassen_2x2_decomposition():
    """Strassen 2x2*2x2 rank-7 decomposition."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _construct_strassen_based_decomposition():
    """Rank-28 decomposition using Strassen on 2x2 blocks."""
    U_s, V_s, W_s = _strassen_2x2_decomposition()
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Product 1: A_left * B11 (terms 0-6)
    for t in range(7):
        if U_s[0][t] != 0: U[0][t] += U_s[0][t]
        if U_s[1][t] != 0: U[1][t] += U_s[1][t]
        if U_s[2][t] != 0: U[4][t] += U_s[2][t]
        if U_s[3][t] != 0: U[5][t] += U_s[3][t]
        if V_s[0][t] != 0: V[0][t] += V_s[0][t]
        if V_s[1][t] != 0: V[1][t] += V_s[1][t]
        if V_s[2][t] != 0: V[4][t] += V_s[2][t]
        if V_s[3][t] != 0: V[5][t] += V_s[3][t]
        if W_s[0][t] != 0: W[0][t] += W_s[0][t]
        if W_s[1][t] != 0: W[2][t] += W_s[1][t]
        if W_s[2][t] != 0: W[1][t] += W_s[2][t]
        if W_s[3][t] != 0: W[3][t] += W_s[3][t]
    
    # Product 2: A_right * B21 (terms 7-13)
    for t in range(7):
        term = t + 7
        if U_s[0][t] != 0: U[2][term] += U_s[0][t]
        if U_s[1][t] != 0: U[3][term] += U_s[1][t]
        if U_s[2][t] != 0: U[6][term] += U_s[2][t]
        if U_s[3][t] != 0: U[7][term] += U_s[3][t]
        if V_s[0][t] != 0: V[8][term] += V_s[0][t]
        if V_s[1][t] != 0: V[9][term] += V_s[1][t]
        if V_s[2][t] != 0: V[12][term] += V_s[2][t]
        if V_s[3][t] != 0: V[13][term] += V_s[3][t]
        if W_s[0][t] != 0: W[0][term] += W_s[0][t]
        if W_s[1][t] != 0: W[2][term] += W_s[1][t]
        if W_s[2][t] != 0: W[1][term] += W_s[2][t]
        if W_s[3][t] != 0: W[3][term] += W_s[3][t]
    
    # Product 3: A_left * B12 (terms 14-20)
    for t in range(7):
        term = t + 14
        if U_s[0][t] != 0: U[0][term] += U_s[0][t]
        if U_s[1][t] != 0: U[1][term] += U_s[1][t]
        if U_s[2][t] != 0: U[4][term] += U_s[2][t]
        if U_s[3][t] != 0: U[5][term] += U_s[3][t]
        if V_s[0][t] != 0: V[2][term] += V_s[0][t]
        if V_s[1][t] != 0: V[3][term] += V_s[1][t]
        if V_s[2][t] != 0: V[6][term] += V_s[2][t]
        if V_s[3][t] != 0: V[7][term] += V_s[3][t]
        if W_s[0][t] != 0: W[4][term] += W_s[0][t]
        if W_s[1][t] != 0: W[6][term] += W_s[1][t]
        if W_s[2][t] != 0: W[5][term] += W_s[2][t]
        if W_s[3][t] != 0: W[7][term] += W_s[3][t]
    
    # Product 4: A_right * B22 (terms 21-27)
    for t in range(7):
        term = t + 21
        if U_s[0][t] != 0: U[2][term] += U_s[0][t]
        if U_s[1][t] != 0: U[3][term] += U_s[1][t]
        if U_s[2][t] != 0: U[6][term] += U_s[2][t]
        if U_s[3][t] != 0: U[7][term] += U_s[3][t]
        if V_s[0][t] != 0: V[10][term] += V_s[0][t]
        if V_s[1][t] != 0: V[11][term] += V_s[1][t]
        if V_s[2][t] != 0: V[14][term] += V_s[2][t]
        if V_s[3][t] != 0: V[15][term] += V_s[3][t]
        if W_s[0][t] != 0: W[4][term] += W_s[0][t]
        if W_s[1][t] != 0: W[6][term] += W_s[1][t]
        if W_s[2][t] != 0: W[5][term] += W_s[2][t]
        if W_s[3][t] != 0: W[7][term] += W_s[3][t]
    
    return {"U": U, "V": V, "W": W}

def _structured_init(n, m, p, r, rng):
    """Structured random initialization with sparse patterns."""
    # Each rank-1 term picks a subset of indices
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for c in range(r):
        # Pick 1-3 non-zero entries in each factor
        nu = rng.randint(1, 3)
        nv = rng.randint(1, 3)
        nw = rng.randint(1, 3)
        
        u_idx = rng.sample(range(n * m), nu)
        v_idx = rng.sample(range(m * p), nv)
        w_idx = rng.sample(range(p * n), nw)
        
        for idx in u_idx:
            U[idx, c] = rng.choice([-1, 1])
        for idx in v_idx:
            V[idx, c] = rng.choice([-1, 1])
        for idx in w_idx:
            W[idx, c] = rng.choice([-1, 1])
    
    return U, V, W

def _als_solve_fixed(U, V, W, n, m, p, r, iters=50, tol=1e-10):
    """
    Alternating least squares: fix two factors, solve for the third.
    For tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    If U,V fixed: for each (k,i), W[k,i,c] should satisfy
    sum_c W[k,i,c] * U[i,j,c]*V[j,k,c] = 1 for all j
    This is a linear system in W[k,i, :].
    """
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    for it in range(iters):
        # Solve for W: for each (k,i), find W[k,i,:] such that
        # sum_c W[k,i,c] * (sum_j U[i,j,c]*V[j,k,c] delta_{j,j}) ... 
        # Actually: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # For fixed U,V: for each (k,i), we need sum_c W[k,i,c] * A_{c} = 1 for all j
        # where A_c = sum_j U[i,j,c] * V[j,k,c]... no wait.
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For fixed U,V and varying W: for each triple (i,j,k), this must equal 1.
        # Fix i,k: sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] = 1 for all j
        # This is: for each j, dot(W[k,i,:], U[i,j,:]*V[j,k,:]) = 1
        # So we have a system: M_{j,c} = U[i,j,c]*V[j,k,c], solve M @ W[k,i,:] = ones(m)
        
        for k in range(p):
            for i in range(n):
                M = U_r[i, :, :] * V_r[:, k, :]  # shape (m, r)
                b = np.ones(m)
                # Solve least squares
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                W_r[k, i, :] = sol
        
        # Solve for U: for each (i,j), find U[i,j,:] such that
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
        # M_{k,c} = V[j,k,c]*W[k,i,c], solve M @ U[i,j,:] = ones(p)
        for i in range(n):
            for j in range(m):
                M = V_r[j, :, :] * W_r[:, i, :]  # shape (p, r)
                b = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                U_r[i, j, :] = sol
        
        # Solve for V: for each (j,k), find V[j,k,:] such that
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
        # M_{i,c} = U[i,j,c]*W[k,i,c], solve M @ V[j,k,:] = ones(n)
        for j in range(m):
            for k in range(p):
                M = U_r[:, j, :] * W_r[k, :, :]  # shape (n, r)
                b = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                V_r[j, k, :] = sol
        
        # Check convergence
        recon = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = np.max(np.abs(recon - 1.0))
        if err < tol:
            break
    
    return U_r.reshape(n*m, r), V_r.reshape(m*p, r), W_r.reshape(p*n, r)

def _round_and_verify(U, V, W, n, m, p):
    """Round to nearest integer and verify."""
    Ur = np.round(U).astype(np.int64)
    Vr = np.round(V).astype(np.int64)
    Wr = np.round(W).astype(np.int64)
    U_list = Ur.tolist()
    V_list = Vr.tolist()
    W_list = Wr.tolist()
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    return None

def _try_rank(r, n, m, p, time_budget, rng):
    """Try to find rank-r decomposition using ALS."""
    start = time.time()
    max_tries = 50
    
    for attempt in range(max_tries):
        if time.time() - start > time_budget:
            break
        
        # Structured random init
        U, V, W = _structured_init(n, m, p, r, rng)
        
        # Run ALS
        U_s, V_s, W_s = _als_solve_fixed(U, V, W, n, m, p, r, iters=30)
        
        # Check float
        if not _verify_float(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, atol=0.1):
            continue
        
        # Round and verify
        result = _round_and_verify(U_s, V_s, W_s, n, m, p)
        if result is not None:
            return result
    
    return None

def _try_sparse_search(r, n, m, p, time_budget, rng):
    """Random search with {-1,0,1} entries."""
    start = time.time()
    max_tries = 2000
    
    for attempt in range(max_tries):
        if time.time() - start > time_budget:
            break
        
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        
        for c in range(r):
            nu = rng.randint(1, 3)
            nv = rng.randint(1, 3)
            nw = rng.randint(1, 3)
            for idx in rng.sample(range(n * m), nu):
                U[idx, c] = rng.choice([-1, 1])
            for idx in rng.sample(range(m * p), nv):
                V[idx, c] = rng.choice([-1, 1])
            for idx in rng.sample(range(p * n), nw):
                W[idx, c] = rng.choice([-1, 1])
        
        if _verify_float(U.tolist(), V.tolist(), W.tolist(), n, m, p, atol=1e-6):
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based rank-28
    strassen_result = _construct_strassen_based_decomposition()
    if _verify_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
        best_rank = 28
        best_result = strassen_result
    
    rng = random.Random(42)
    
    # Search for lower ranks
    for r in range(16, best_rank):
        elapsed = time.time() - start_time
        remaining = HARD_LIMIT - elapsed
        if remaining < 5:
            break
        
        # Allocate time between ALS and sparse search
        time_als = remaining * 0.6
        time_sparse = remaining * 0.3
        
        # Try ALS-based search
        result = _try_rank(r, n, m, p, time_als, rng)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
        
        # Try sparse random search
        elapsed = time.time() - start_time
        remaining = HARD_LIMIT - elapsed
        if remaining < 3:
            break
        
        result = _try_sparse_search(r, n, m, p, time_sparse, rng)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return best_result
