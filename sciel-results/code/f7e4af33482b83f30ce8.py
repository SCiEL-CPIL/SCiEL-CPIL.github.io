import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def check_decomposition(U, V, W, tol=1e-10):
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.sum(W[k*n+i] * U[i*m+j] * V[jp*p+k])
                        expected = 1.0 if j == jp else 0.0
                        if abs(val - expected) > tol:
                            return False
        return True
    
    def try_rank(r, seed=42, max_iters=3000):
        if time.time() > deadline - 40:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        for iteration in range(max_iters):
            if time.time() > deadline - 40:
                return None
            
            # Update W
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m*m, r))
                    b_vec = np.zeros(m*m)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            A_mat[idx, :] = U[i*m+j] * V[jp*p+k]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Update U
            for i in range(n):
                A_mat = np.zeros((p*m*m, r))
                b_vec = np.zeros(p*m*m)
                idx = 0
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            A_mat[idx, :] = W[k*n+i] * V[jp*p+k]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    U[i*m:(i+1)*m, :] = sol.reshape(m, r)
                except:
                    pass
            
            # Update V
            for k in range(p):
                A_mat = np.zeros((n*m*m, r))
                b_vec = np.zeros(n*m*m)
                idx = 0
                for i in range(n):
                    for jp in range(m):
                        for j in range(m):
                            A_mat[idx, :] = W[k*n+i] * U[i*m+j]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    V[:, k] = sol.reshape(m, r).T
                except:
                    pass
            
            # Check
            max_err = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = np.sum(W[k*n+i] * U[i*m+j] * V[jp*p+k])
                            expected = 1.0 if j == jp else 0.0
                            max_err = max(max_err, abs(val - expected))
            if max_err < 1e-10:
                if check_decomposition(U, V, W, tol=1e-10):
                    return U, V, W
        
        return None
    
    # Try ranks
    for r in range(30, 84):
        if time.time() > deadline - 60:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 40:
                break
            result = try_rank(r, seed=seed, max_iters=2000)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W, best_rank = U, V, W, r
                found = True
                break
        if found:
            break
    
    def to_list(mat):
        lst = []
        for row in mat:
            lst_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) < 1e-9:
                    lst_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-9:
                    if nearest_half == int(nearest_half):
                        lst_row.append(int(nearest_half))
                    else:
                        lst_row.append(float(nearest_half))
                else:
                    lst_row.append(float(val))
            lst.append(lst_row)
        return lst
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
