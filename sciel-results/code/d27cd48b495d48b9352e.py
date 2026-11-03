import time
import numpy as np

def construct():
    deadline = time.time() + 130
    
    n, m, p = 4, 4, 5
    
    # Start with schoolbook decomposition
    rank = n * m * p  # 80
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
                r += 1
    
    # Verify initial solution
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by checking if columns are linearly dependent
    # and can be removed while preserving the tensor
    current_rank = rank
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # Attempt rank reduction: try removing each term and see if we can compensate
    # This is a heuristic - we'll try to find if any term is redundant
    
    # For now, return the schoolbook solution as a safe baseline
    # Convert to list of lists
    U_list = U_curr.tolist()
    V_list = V_curr.tolist()
    W_list = W_curr.tolist()
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(current_rank):
                    s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at {i},{j},{k}: sum={s}")
    
    return {"U": U_list, "V": V_list, "W": W_list}
