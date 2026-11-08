import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline to be safe
    
    # Try to find a rank < 40 solution (e.g. 35-39)
    # Schoolbook is 40. Strassen blocks suggest 38 is possible.
    for rank in range(35, 41):
        if time.time() > deadline:
            break
        result = try_rank(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank(n, m, p, rank):
    # Try a few seeds, but limit iterations to stay within time
    for seed in range(5):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for iteration in range(50):
            # Solve for U
            U_new = np.zeros((n*m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for s in range(rank):
                            A[k, s] = V[j*p+k, s] * W[k*n+i, s]
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        return None
            U = U_new
            
            # Solve for V
            V_new = np.zeros((m*p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for s in range(rank):
                            A[i, s] = U[i*m+j, s] * W[k*n+i, s]
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        return None
            V = V_new
            
            # Solve for W
            W_new = np.zeros((p*n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for s in range(rank):
                            A[j, s] = U[i*m+j, s] * V[j*p+k, s]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        return None
            W = W_new
            
            # Check convergence
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(rank))
                        err += (val - 1)**2
            if err < 1e-10:
                # Round to nearest integer
                U_int = np.rint(U)
                V_int = np.rint(V)
                W_int = np.rint(W)
                
                # Verify exactness with integers
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_int[i*m+j, s] * V_int[j*p+k, s] * W_int[k*n+i, s] for s in range(rank))
                            if val != 1:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return None
