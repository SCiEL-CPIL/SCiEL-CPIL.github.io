import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_schoolbook = n * m * p
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0

    # Schoolbook solution
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U_sb[u, col] = 1.0
                V_sb[v, col] = 1.0
                W_sb[w, col] = 1.0
                col += 1

    def check_solution(U, V, W):
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_check, atol=1e-9) and
                np.allclose(V, V_check, atol=1e-9) and
                np.allclose(W, W_check, atol=1e-9)):
            return False
            
        T_decomp = np.einsum('ur,vr,wr->uvw', U_check, V_check, W_check)
        if np.allclose(T_decomp, T, atol=1e-9):
            return True
        return False

    def convert_to_list(mat):
        return mat.tolist()

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    time_limit = 140.0
    
    # Try to find rank 49
    target_rank = rank_schoolbook - 1
    
    if time.time() - start_time < time_limit:
        # Objective function
        def objective(x):
            U = x[:nm*target_rank].reshape(nm, target_rank)
            V = x[nm*target_rank:(nm+mp)*target_rank].reshape(mp, target_rank)
            W = x[(nm+mp)*target_rank:].reshape(pn, target_rank)
            
            T_decomp = np.einsum('ur,vr,wr->uvw', U, V, W)
            error = np.sum((T_decomp - T)**2)
            return error

        # Initial guess: schoolbook solution with one column removed?
        # Or random initialization
        # Let's try random initialization with small values
        np.random.seed(42)
        x0 = np.random.randn(nm*target_rank + mp*target_rank + pn*target_rank) * 0.1
        
        # Run optimization
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-8})
        
        if res.fun < 1e-6:
            U_opt = res.x[:nm*target_rank].reshape(nm, target_rank)
            V_opt = res.x[nm*target_rank:(nm+mp)*target_rank].reshape(mp, target_rank)
            W_opt = res.x[(nm+mp)*target_rank:].reshape(pn, target_rank)
            
            if check_solution(U_opt, V_opt, W_opt):
                best_rank = target_rank
                best_U, best_V, best_W = U_opt, V_opt, W_opt

    return {"U": convert_to_list(best_U), "V": convert_to_list(best_V), "W": convert_to_list(best_W)}
