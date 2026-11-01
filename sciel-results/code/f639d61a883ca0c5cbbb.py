import numpy as np
import time
import random
from scipy.optimize import minimize, basinhopping

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Target ranks to try, starting from lower bound
    # Lower bound is max(n*m, m*p, p*n) = 6
    # Known rank for 2x2x3 is 10
    target_ranks = [10, 11]

    # Time management
    start_time = time.time()
    deadline = start_time + 140.0 # 140s limit to be safe (150s kill)

    def get_tensor(U, V, W):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Returns the full tensor T'_{a,b,c}
        """
        rank = U.shape[1]
        T_prime = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_r = np.array(U[:, r])
            v_r = np.array(V[:, r])
            w_r = np.array(W[:, r])
            T_prime += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        return T_prime

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()

    def objective(params):
        if time.time() > deadline:
            return 1e9
        rank = params.shape[0] // (n*m + m*p + p*n)
        if rank == 0:
            return 1e9
            
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        T_prime = get_tensor(U, V, W)
        diff = T_prime - T_target
        return np.sum(diff**2)

    def try_find_decomposition(target_rank):
        if time.time() > deadline - 10.0:
            return None

        random.seed(42 + target_rank)
        np.random.seed(42 + target_rank)
        
        init_size = (n*m + m*p + p*n) * target_rank
        
        # Try a few random initializations with basinhopping
        for attempt in range(5): # 5 basinhopping runs
            if time.time() > deadline - 5.0:
                break
                
            # Initialize with small random values
            params0 = np.random.randn(init_size) * 0.5
            
            # Local minimization first
            try:
                res_local = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
                if res_local.fun < 1e-6:
                    # Verify and return
                    U_flat = res_local.x[:n*m*target_rank]
                    V_flat = res_local.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = res_local.x[n*m*target_rank + m*p*target_rank:]
                    
                    U_num = U_flat.reshape(n*m, target_rank)
                    V_num = V_flat.reshape(m*p, target_rank)
                    W_num = W_flat.reshape(p*n, target_rank)
                    
                    # Round to nearest 0.5
                    U_rnd = np.round(U_num * 2) / 2
                    V_rnd = np.round(V_num * 2) / 2
                    W_rnd = np.round(W_num * 2) / 2
                    
                    # Verify exactness
                    T_rnd = get_tensor(U_rnd, V_rnd, W_rnd)
                    if np.allclose(T_rnd, T_target, atol=1e-9):
                        return U_rnd, V_rnd, W_rnd
            except Exception:
                pass

            # Global search with basinhopping
            try:
                def step(x):
                    # Random kick
                    return x + np.random.randn(len(x)) * 0.1
                
                res_hop = basinhopping(objective, params0, niter=20, stepsize=0.5, 
                                        minimizer_kwargs={"method": "L-BFGS-B", "options": {"maxiter": 50}},
                                        stepfunc=step)
                
                if res_hop.fun < 1e-6:
                    U_flat = res_hop.x[:n*m*target_rank]
                    V_flat = res_hop.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = res_hop.x[n*m*target_rank + m*p*target_rank:]
                    
                    U_num = U_flat.reshape(n*m, target_rank)
                    V_num = V_flat.reshape(m*p, target_rank)
                    W_num = W_flat.reshape(p*n, target_rank)
                    
                    # Round to nearest 0.5
                    U_rnd = np.round(U_num * 2) / 2
                    V_rnd = np.round(V_num * 2) / 2
                    W_rnd = np.round(W_num * 2) / 2
                    
                    # Verify exactness
                    T_rnd = get_tensor(U_rnd, V_rnd, W_rnd)
                    if np.allclose(T_rnd, T_target, atol=1e-9):
                        return U_rnd, V_rnd, W_rnd
            except Exception:
                pass

        return None

    for rank in target_ranks:
        if time.time() > deadline - 10.0:
            break
        result = try_find_decomposition(rank)
        if result is not None:
            U, V, W = result
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            best_rank = rank
            break

    return {"U": best_U, "V": best_V, "W": best_W}
