import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 7
    
    # Try to find rank 24 decomposition
    R = 24
    # Variables: U (4xR), V (14xR), W (14xR)
    # Total 32*R variables
    
    def residuals(x):
        U = x[:4*R].reshape(4, R)
        V = x[4*R:4*R+14*R].reshape(14, R)
        W = x[4*R+14*R:].reshape(14, R)
        
        res = np.zeros(28)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(7):
                    s = 0.0
                    for r in range(R):
                        s += U[i, r] * V[j*7+k, r] * W[k*2+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    # Initialization: Use the rank-25 solution and drop one term? 
    # Or random.
    # Random might work if we run multiple times.
    
    best_x = None
    best_res = None
    
    # Time budget
    start_time = time.time()
    timeout = 100 # Leave 20s for fallback/verification
    
    for attempt in range(5):
        if time.time() - start_time > timeout:
            break
            
        # Initialize with small random values
        np.random.seed(attempt)
        x0 = np.random.randn(32*R) * 0.5
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if sol.cost < best_res or best_res is None:
                best_res = sol.cost
                best_x = sol.x
        except Exception:
            pass
    
    if best_x is not None and best_res < 1e-6:
        U = best_x[:4*R].reshape(4, R)
        V = best_x[4*R:4*R+14*R].reshape(14, R)
        W = best_x[4*R+14*R:].reshape(14, R)
        
        # Try to round to half-integers
        # Check if rounding works
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(7):
                    s = 0.0
                    for r in range(R):
                        s += U_rounded[i, r] * V_rounded[j*7+k, r] * W_rounded[k*2+i, r]
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
    # Fallback to rank 25 (current implementation)
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)

    col_start = 21
    k = 6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1

    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1

    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1

    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
