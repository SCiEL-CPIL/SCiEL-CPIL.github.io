"""Find rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def check_solution(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0)**2
        return err
    
    def alternating_minimization(r, max_iter=300, tol=1e-12, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = sol
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
                    sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = sol
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                    sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = sol
            
            err = compute_error(U_new, V_new, W_new)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            if err < tol:
                break
            
            U, V, W = U_new, V_new, W_new
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from lower bound up
    best_rank = rank_sb
    best_solution = None
    
    for r in range(16, rank_sb):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(30):
            if time.time() > deadline:
                break
            
            U, V, W, err = alternating_minimization(r, max_iter=100, tol=1e-12, seed=seed)
            
            if err < 1e-8:
                if check_solution(U, V, W, tol=1e-6):
                    found = True
                    best_rank = r
                    best_solution = (U, V, W)
                    break
        
        if found:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        U_list = [[float(U[i,j]) for j in range(best_rank)] for i in range(n*m)]
        V_list = [[float(V[i,j]) for j in range(best_rank)] for i in range(m*p)]
        W_list = [[float(W[i,j]) for j in range(best_rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(U_sb[i,j]) for j in range(rank_sb)] for i in range(n*m)]
        V_list = [[float(V_sb[i,j]) for j in range(rank_sb)] for i in range(m*p)]
        W_list = [[float(W_sb[i,j]) for j in range(rank_sb)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
