"""Matrix multiplication tensor decomposition for 4x5 * 5x6."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Schoolbook decomposition (rank = 120)
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    def verify(U_list, V_list, W_list, tol=1e-9):
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return np.sqrt(err)
    
    def try_decomposition(rank, max_iters=3000, tol=1e-8, seed=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U_curr = np.random.randn(n * m, rank) * 0.5
        V_curr = np.random.randn(m * p, rank) * 0.5
        W_curr = np.random.randn(p * n, rank) * 0.5
        
        best_error = np.inf
        best_solution = None
        stagnation = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = U_curr.copy()
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V_curr[j * p + k, :] * W_curr[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = x
                    except:
                        pass
            
            # Update V
            V_new = V_curr.copy()
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U_new[i * m + j, :] * W_curr[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = x
                    except:
                        pass
            
            # Update W
            W_new = W_curr.copy()
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = x
                    except:
                        pass
            
            error = compute_error(U_new, V_new, W_new)
            
            if error < best_error:
                best_error = error
                best_solution = (U_new.copy(), V_new.copy(), W_new.copy())
                stagnation = 0
            else:
                stagnation += 1
            
            if error < tol:
                break
            
            if stagnation > 100:
                break
            
            U_curr = U_new
            V_curr = V_new
            W_curr = W_new
        
        if best_error < tol and best_solution is not None:
            U_sol, V_sol, W_sol = best_solution
            U_rounded = np.round(U_sol * 2) / 2
            V_rounded = np.round(V_sol * 2) / 2
            W_rounded = np.round(W_sol * 2) / 2
            
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Try ranks
    for rank in [115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]:
        if time.time() > deadline:
            break
        result = try_decomposition(rank, max_iters=2000, tol=1e-8)
        if result is not None:
            best_result = result
            break
    
    return best_result
