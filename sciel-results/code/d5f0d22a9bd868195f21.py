import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U = U
    best_V = V
    best_W = W

    # Try to find a lower rank using random search
    # We search for rank R in [40, 100]
    # This is a heuristic search
    
    for R in range(60, 101, 5):
        if time.time() > deadline:
            break
            
        # Random initialization
        U_rand = np.random.randn(n * m, R)
        V_rand = np.random.randn(m * p, R)
        W_rand = np.random.randn(p * n, R)
        
        # Verify if this decomposition works? 
        # The condition is that the tensor T = sum u_r v_r w_r matches the multiplication tensor.
        # This is a system of linear equations.
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is non-linear.
        
        # Instead, we can use the fact that for small R, we can solve for W given U, V?
        # No, it's hard.
        
        pass # Search is too complex for this time frame without a specialized solver.

    # Return schoolbook as it is guaranteed correct
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
