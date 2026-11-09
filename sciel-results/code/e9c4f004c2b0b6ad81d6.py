"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = result
    best_rank = rank
    
    # Precompute index tuples for efficient access
    idx_list = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try ranks from 35 down to 28
    for r in range(35, 27, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations with alternating least squares
        for restart in range(25):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + r * 100 + restart)
            U_rand = np.random.randn(n*m, r)
            V_rand = np.random.randn(m*p, r)
            W_rand = np.random.randn(p*n, r)
            
            # Alternating least squares: fix two, solve for third
            for it in range(150):
                if time.time() > deadline - 30:
                    break
                
                # Fix V and W, solve for U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V_rand[j*p+k] * W_rand[k*n+i] for k in range(p)]).T  # shape (r, p)
                        b = np.ones(p)
                        try:
                            U_rand[i*m+j] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U_rand[i*m+j] * W_rand[k*n+i] for i in range(n)]).T  # shape (r, n)
                        b = np.ones(n)
                        try:
                            V_rand[j*p+k] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U_rand[i*m+j] * V_rand[j*p+k] for j in range(m)]).T  # shape (r, m)
                        b = np.ones(m)
                        try:
                            W_rand[k*n+i] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            pass
            
            # Round to half-integers
            U_rounded = np.round(U_rand * 2) / 2
            V_rounded = np.round(V_rand * 2) / 2
            W_rounded = np.round(W_rand * 2) / 2
            
            # Check if exact using numpy
            exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                        if abs(s - 1) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                result = {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                best_result = result
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
