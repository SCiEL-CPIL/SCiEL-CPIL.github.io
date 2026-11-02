import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target: Find rank minimal decomposition for n x m * m x p
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def make_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T

    def verify(U, V, W, T, tol=1e-6):
        R = U.shape[1]
        try:
            Temp = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', Temp, W)
            return np.allclose(T_approx, T, atol=tol)
        except:
            return False

    def try_find(n, m, p, target_rank, time_limit):
        T = make_tensor(n, m, p)
        nm, mp, pn = n*m, m*p, p*n
        best_local = None
        
        start_time = time.time()
        max_attempts = 200
        attempt_time = time_limit / max_attempts if max_attempts > 0 else time_limit
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            # Try small integers
            U = np.random.randint(-2, 3, (nm, target_rank))
            V = np.random.randint(-2, 3, (mp, target_rank))
            W = np.random.randint(-2, 3, (pn, target_rank))
            
            if verify(U, V, W, T, tol=1e-9):
                return U, V, W
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_x = x[:nm*target_rank].reshape(nm, target_rank)
                V_x = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_x = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                try:
                    Temp = np.einsum('ur,vr->uvr', U_x, V_x)
                    T_approx = np.einsum('uvr,wr->uvw', Temp, W_x)
                    err = np.sum((T_approx - T)**2)
                    return err
                except:
                    return 1e10

            # Use Powell method which is derivative-free and often better for these problems
            res = minimize(objective, x0, method='Powell', options={'maxiter': 50, 'ftol': 1e-12})
            
            if res.fun < 1e-4:
                U_opt = res.x[:nm*target_rank].reshape(nm, target_rank)
                V_opt = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_opt = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round, T, tol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None

    time_limit_total = 100
    t_start = time.time()
    
    # Try rank 18 directly
    # LB is 18.
    res_direct = try_find(2, 3, 6, 18, 50)
    if res_direct:
        U_d, V_d, W_d = res_direct
        return {
            "U": U_d,
            "V": V_d,
            "W": W_d
        }
        
    # If 18 fails, try 19
    if time.time() - t_start < 80:
        res_direct = try_find(2, 3, 6, 19, 20)
        if res_direct:
            U_d, V_d, W_d = res_direct
            return {
                "U": U_d,
                "V": V_d,
                "W": W_d
            }

    # Fallback: Schoolbook
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
