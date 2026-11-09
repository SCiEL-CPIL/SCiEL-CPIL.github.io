import time
import numpy as np
from scipy.optimize import fsolve

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = get_schoolbook()
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if not np.isclose(np.sum(prod), 1.0, atol=1e-6):
                        return False
        return True
    
    if not verify(U_sb, V_sb, W_sb):
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    for target_rank in range(rank_sb - 1, max(rank_sb - 10, 30), -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(42 + attempt + target_rank)
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Define the system of equations
            def equations(vars):
                U = vars[:n*m*target_rank].reshape(n*m, target_rank)
                V = vars[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = vars[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                eqs = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                            eqs.append(np.sum(prod) - 1.0)
                return np.array(eqs)
            
            initial_vars = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                solution = fsolve(equations, initial_vars, full_output=True)
                sol, info, ier, msg = solution
                
                if ier == 1:
                    U_sol = sol[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = sol[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = sol[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    if verify(U_sol, V_sol, W_sol):
                        # Try to round to half-integers
                        U_round = np.round(U_sol * 2) / 2
                        V_round = np.round(V_sol * 2) / 2
                        W_round = np.round(W_sol * 2) / 2
                        
                        if verify(U_round, V_round, W_round):
                            best_U, best_V, best_W = U_round, V_round, W_round
                            best_rank = target_rank
                            found = True
                            break
            except Exception:
                continue
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
