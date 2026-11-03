import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Lower bound for rank
    # Trivial lower bound: max(n*m, m*p, p*n) = 12
    # But for matrix multiplication, it's known to be higher.
    # We start from a reasonable lower bound, say 20, to save time.
    # Actually, let's start from 15 as in the original, but optimize the loop.
    
    start_rank = 15
    max_rank = n * m * p # 36
    
    # Pre-compute indices for efficiency
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def verify(U, V, W, r):
        # Vectorized verification
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is not a simple matrix product due to the different index mappings.
        # We can use einsum or loop.
        # Loop is 36 * r operations. For r=20, 720 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(arr, tol=1e-6):
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Try ranks from low to high
    for r in range(start_rank, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Number of attempts increases for lower ranks
        # If r is close to max_rank, fewer attempts needed
        attempts = 50 if r < 25 else 20
        
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize V and W with small random integers
            # Using -1, 0, 1 helps with finding sparse/rational solutions
            V = np.random.randint(-1, 2, size=(m * p, r)).astype(float)
            W = np.random.randint(-1, 2, size=(p * n, r)).astype(float)
            
            # Ensure not all zeros
            if np.all(V == 0): V[0,0] = 1
            if np.all(W == 0): W[0,0] = 1
            
            U = np.zeros((n * m, r))
            valid = True
            
            # ALS Steps
            for step in range(10):
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_ij = np.zeros((p, r))
                        for k in range(p):
                            A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        b = np.ones(p)
                        try:
                            u, residuals, rank, s = np.linalg.lstsq(A_ij, b, rcond=None)
                            if np.allclose(A_ij @ u, b, atol=1e-9):
                                U[i * m + j, :] = u
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros((n, r))
                        for i in range(n):
                            B_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        b = np.ones(n)
                        try:
                            v, residuals, rank, s = np.linalg.lstsq(B_jk, b, rcond=None)
                            if np.allclose(B_jk @ v, b, atol=1e-9):
                                V[j * p + k, :] = v
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        C_ik = np.zeros((m, r))
                        for j in range(m):
                            C_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        b = np.ones(m)
                        try:
                            w, residuals, rank, s = np.linalg.lstsq(C_ik, b, rcond=None)
                            if np.allclose(C_ik @ w, b, atol=1e-9):
                                W[k * n + i, :] = w
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                U_rounded = is_half_integer(U)
                V_rounded = is_half_integer(V)
                W_rounded = is_half_integer(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if verify(U_rounded, V_rounded, W_rounded, r):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
    # Fallback: Schoolbook algorithm
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
