import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def solve_for_W(U, V, r):
        W = np.zeros((pn, r))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                if np.linalg.matrix_rank(A) >= m:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                else:
                    x = np.zeros(r)
                    # Fallback: distribute evenly
                    if np.any(np.abs(A) > 1e-10):
                        x = np.ones(r) / r
                    else:
                        x = np.ones(r)
                W[k*n+i, :] = x
        return W
    
    def solve_for_U(V, W, r):
        U = np.zeros((nm, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                if np.linalg.matrix_rank(A) >= p:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                else:
                    x = np.ones(r)
                U[i*m+j, :] = x
        return U
    
    def solve_for_V(U, W, r):
        V = np.zeros((mp, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                if np.linalg.matrix_rank(A) >= n:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                else:
                    x = np.ones(r)
                V[j*p+k, :] = x
        return V
    
    def check_residual(U, V, W, r):
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res = abs(val - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res
    
    def to_half_integer(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2
    
    def try_rank(r, num_restarts=20):
        best_result = None
        best_res = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 12345 + r)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                W = solve_for_W(U, V, r)
                U = solve_for_U(V, W, r)
                V = solve_for_V(U, W, r)
                
                res = check_residual(U, V, W, r)
                if res < 1e-12:
                    # Try to round to half-integers
                    U_hi = to_half_integer(U)
                    V_hi = to_half_integer(V)
                    W_hi = to_half_integer(W)
                    res_hi = check_residual(U_hi, V_hi, W_hi, r)
                    if res_hi < 1e-9:
                        return U_hi, V_hi, W_hi, res_hi
                    best_result = (U, V, W)
                    best_res = res
                    break
            
            res = check_residual(U, V, W, r)
            if res < best_res:
                best_res = res
                best_result = (U, V, W)
        
        if best_result is not None and best_res < 1e-9:
            U, V, W = best_result
            U_hi = to_half_integer(U)
            V_hi = to_half_integer(V)
            W_hi = to_half_integer(W)
            res_hi = check_residual(U_hi, V_hi, W_hi, r)
            if res_hi < 1e-9:
                return U_hi, V_hi, W_hi, res_hi
        
        return None
    
    # Try ranks from 19 to 23
    for r in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r, num_restarts=15)
        if result is not None:
            U, V, W, _ = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
