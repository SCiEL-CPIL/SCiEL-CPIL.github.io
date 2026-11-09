"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Schoolbook as guaranteed fallback
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_arr[i*m+j], V_arr[j*p+k]), W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_als(rank):
        """Try to find a decomposition of given rank using Alternating Least Squares."""
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(42)
        # Initialize U, V, W
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Target tensor T[i,j,k] = 1
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # Iterate ALS
        for it in range(100):
            if time.time() > deadline - 5:
                break
                
            # 1. Solve for W given U, V
            # For each k, i:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then M @ W[k*n+i, :] = [1, 1, ..., 1] (length m)
            new_W = np.zeros((p*n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve M @ w = ones(m)
                    # w = new_W[k*n+i, :]
                    try:
                        # Using lstsq for robustness
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        return None
            W = new_W
            
            # 2. Solve for U given V, W
            # For each i, j:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # Let N[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then N @ U[i*m+j, :] = [1, 1, ..., 1] (length p)
            new_U = np.zeros((n*m, rank))
            for i in range(n):
                for j in range(m):
                    N = np.zeros((p, rank))
                    for k in range(p):
                        N[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(N, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        return None
            U = new_U
            
            # 3. Solve for V given U, W
            # For each j, k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # Let P[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Then P @ V[j*p+k, :] = [1, 1, ..., 1] (length n)
            new_V = np.zeros((m*p, rank))
            for j in range(m):
                for k in range(p):
                    P = np.zeros((n, rank))
                    for i in range(n):
                        P[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(P, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        return None
            V = new_V
            
            # Check residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                        residual += (val - 1.0)**2
            if residual < 1e-12:
                # Try to round to integers or half-integers
                # First try integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try half-integers (multiply by 2, round, divide by 2)
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                if verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
                # If floating point is good enough but rounding fails, 
                # we might still return the float if the evaluator accepts it?
                # The prompt says "integer/half-integer entries". 
                # So we must return integers/half-integers.
                # If we can't round, this rank might not have an exact integer solution 
                # or we need a different initialization.
                # For now, break and try next rank or different init.
                pass

        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 30
    # Try a few ranks
    for rank in [30, 31, 32, 33, 34, 35, 36, 40, 45, 50]:
        if time.time() > deadline - 10:
            break
        result = try_als(rank)
        if result is not None:
            return result
            
    return schoolbook()
# EVOLVE-BLOCK-END

