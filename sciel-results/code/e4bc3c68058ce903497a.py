import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_2arr = np.zeros((p * n, rank), dtype=np.int64)
    for k in range(p):
        for i in range(n):
            row = W[k * n + i]
            for c in range(rank):
                val = row[c]
                if isinstance(val, Frac):
                    W_2arr[k * n + i, c] = int(val * 2)
                else:
                    W_2arr[k * n + i, c] = int(val) * 2

    for i in range(n):
        for k in range(p):
            U_sub = U_arr[i * m : (i + 1) * m, :]
            V_sub = V_arr[np.arange(m) * p + k, :]
            W_vec = W_2arr[k * n + i, :]
            
            A = U_sub * W_vec
            result = A @ V_sub.T
            expected = 2 * np.eye(m, dtype=np.int64)
            
            if not np.array_equal(result, expected):
                return False
    return True

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    range_val = 2 # -2 to 2, smaller range increases hit rate for small ranks
    
    while time.time() < deadline and it < max_iter:
        it += 1
        U = rng.randint(-range_val, range_val, (nm, rank))
        V = rng.randint(-range_val, range_val, (mp, rank))
        
        if np.any(np.all(U == 0, axis=1)):
            continue
        if np.any(np.all(V == 0, axis=1)):
            continue
        
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                U_sub = U[i * m : (i + 1) * m, :]
                V_sub = V[np.arange(m) * p + k, :]
                
                M = (U_sub[:, np.newaxis, :] * V_sub[np.newaxis, :, :]).reshape(m * m, rank)
                
                b = np.zeros(m * m, dtype=float)
                for j in range(m):
                    b[j * m + j] = 1.0
                
                try:
                    W_float = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                
                W_half_int_float = np.round(W_float * 2) / 2
                if not np.allclose(W_float, W_half_int_float, atol=1e-9):
                    ok = False
                    break
                
                W_vec = []
                for val in W_half_int_float:
                    k_val = int(round(val * 2))
                    W_vec.append(Frac(k_val, 2))
                
                W_2_vec = np.array([int(w * 2) for w in W_vec], dtype=np.int64)
                A = U_sub * W_2_vec
                result = A @ V_sub.T
                expected = 2 * np.eye(m, dtype=np.int64)
                
                if not np.array_equal(result, expected):
                    ok = False
                    break
                
                W[k * n + i] = W_vec
            if not ok:
                break
        
        if not ok:
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U.tolist(), V.tolist(), W
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    best_result = _schoolbook(n, m, p)
    
    # Try ranks in increasing order
    # Rank 26 is the theoretical minimum for 3x3x4.
    target_ranks = [26, 27, 28, 29, 30]
    
    for rank in target_ranks:
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(rank * 12345 + 678)
        start_rank_time = time.time()
        rank_deadline = min(deadline - 2, start_rank_time + 20) # 20s per rank
        
        try:
            result = _search_rank(rank, n, m, p, rank_deadline, rng, max_iter=500000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                if rank <= 27:
                    break
        except Exception:
            pass

    return best_result
