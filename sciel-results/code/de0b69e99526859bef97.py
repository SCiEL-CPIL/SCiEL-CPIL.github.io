import time
import numpy as np
import random

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def _verify(U, V, W, n, m, p, R):
    """Verify that sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 for all i,j,k"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(R):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _try_solve_for_U(V, W, n, m, p, R):
    """Given V and W, solve for U using linear systems per j."""
    U = np.zeros((n * m, R))
    for j in range(m):
        M = np.zeros((n * p, n * R))
        b = np.ones(n * p)
        
        for i in range(n):
            for k in range(p):
                row_idx = i * p + k
                for t in range(R):
                    col_idx = i * R + t
                    M[row_idx, col_idx] = V[j * p + k, t] * W[k * n + i, t]
        
        try:
            u, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        u_matrix = u.reshape((n, R))
        
        # Try rounding strategies
        best_u = None
        best_err = float('inf')
        
        # Strategy 1: Round to integers
        u_int = np.round(u_matrix).astype(float)
        err1 = 0.0
        for i in range(n):
            for k in range(p):
                val = sum(u_int[i, t] * V[j * p + k, t] * W[k * n + i, t] for t in range(R))
                err1 += (val - 1) ** 2
        if err1 < best_err:
            best_err = err1
            best_u = u_int.copy()
        
        # Strategy 2: Round to half-integers
        u_half = np.round(u_matrix * 2) / 2.0
        err2 = 0.0
        for i in range(n):
            for k in range(p):
                val = sum(u_half[i, t] * V[j * p + k, t] * W[k * n + i, t] for t in range(R))
                err2 += (val - 1) ** 2
        if err2 < best_err:
            best_err = err2
            best_u = u_half.copy()
        
        # Strategy 3: Local rounding search for each entry
        for i in range(n):
            for t in range(R):
                base = u_matrix[i, t]
                candidates = [
                    float(int(np.floor(base))),
                    float(int(np.ceil(base))),
                    float(int(np.floor(base * 2)) / 2.0),
                    float(int(np.ceil(base * 2)) / 2.0),
                ]
                for c in candidates:
                    if abs(c - base) > 2:
                        continue
                    test_u = u_matrix.copy()
                    test_u[i, t] = c
                    err_test = 0.0
                    for ii in range(n):
                        for k in range(p):
                            val = sum(test_u[ii, t2] * V[j * p + k, t2] * W[k * n + ii, t2] for t2 in range(R))
                            err_test += (val - 1) ** 2
                    if err_test < best_err:
                        best_err = err_test
                        best_u = test_u.copy()
        
        if best_err < 1e-9:
            for i in range(n):
                U[i * m + j, :] = best_u[i, :]
        else:
            return None
    
    return U

def _local_search(V, W, n, m, p, R, max_iters=20):
    """Local search: perturb V or W entries and re-solve for U."""
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 12:
            break
        
        if random.random() < 0.5:
            ri, rt = random.randint(0, m * p - 1), random.randint(0, R - 1)
            old_val = best_V[ri, rt]
            for delta in [-1, 1]:
                best_V[ri, rt] = old_val + delta
                U = _try_solve_for_U(best_V, best_W, n, m, p, R)
                if U is not None and _verify(U, best_V, best_W, n, m, p, R):
                    return U, best_V.copy(), best_W.copy()
            best_V[ri, rt] = old_val
        else:
            ri, rt = random.randint(0, p * n - 1), random.randint(0, R - 1)
            old_val = best_W[ri, rt]
            for delta in [-1, 1]:
                best_W[ri, rt] = old_val + delta
                U = _try_solve_for_U(best_V, best_W, n, m, p, R)
                if U is not None and _verify(U, best_V, best_W, n, m, p, R):
                    return U, best_V.copy(), best_W.copy()
            best_W[ri, rt] = old_val
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    for R in range(30, best_rank):
        if time.time() > DEADLINE - 15:
            break
        
        attempts_for_rank = 0
        
        for attempt in range(3000):
            if time.time() > DEADLINE - 10:
                break
            
            attempts_for_rank += 1
            init_type = attempt % 5
            
            if init_type == 0:
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for _ in range(int(R * 1.5)):
                    ri = random.randint(0, m * p - 1)
                    rt = random.randint(0, R - 1)
                    V[ri, rt] = random.choice([-1, 1])
                for _ in range(int(R * 1.5)):
                    ri = random.randint(0, p * n - 1)
                    rt = random.randint(0, R - 1)
                    W[ri, rt] = random.choice([-1, 1])
            elif init_type == 1:
                V = np.random.randint(-1, 2, size=(m * p, R))
                W = np.random.randint(-1, 2, size=(p * n, R))
            elif init_type == 2:
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for t in range(R):
                    v_pos = random.randint(0, m * p - 1)
                    w_pos = random.randint(0, p * n - 1)
                    V[v_pos, t] = random.choice([-1, 1])
                    W[w_pos, t] = random.choice([-1, 1])
            elif init_type == 3:
                V = np.random.randint(-2, 3, size=(m * p, R))
                W = np.random.randint(-2, 3, size=(p * n, R))
                mask = np.random.random(V.shape) < 0.4
                V[mask] = 0
                mask = np.random.random(W.shape) < 0.4
                W[mask] = 0
            else:
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for t in range(R):
                    num_nonzero = random.randint(2, 3)
                    for _ in range(num_nonzero):
                        ri = random.randint(0, m * p - 1)
                        V[ri, t] = random.choice([-1, 1])
                    for _ in range(num_nonzero):
                        ri = random.randint(0, p * n - 1)
                        W[ri, t] = random.choice([-1, 1])
            
            U = _try_solve_for_U(V, W, n, m, p, R)
            
            if U is not None:
                if _verify(U, V, W, n, m, p, R):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    U_ls, V_ls, W_ls = _local_search(V, W, n, m, p, R, max_iters=15)
                    if U_ls is not None:
                        return {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
        
        if attempts_for_rank >= 500 and time.time() > DEADLINE - 30:
            break
    
    return best_result
