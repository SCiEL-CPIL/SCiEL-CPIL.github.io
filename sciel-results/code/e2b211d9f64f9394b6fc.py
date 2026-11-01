import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=9

    # Trivial decomposition fallback (Rank 36)
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # Build rank-32 decomposition
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1; V[k0][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[2][col] = 1; U[3][col] = 1; V[k0][col] = 1; W[2*k0+1][col] = 1; W[2*k1+1][col] = -1; col += 1
            U[0][col] = 1; V[k1][col] = 1; V[p+k1][col] = -1; W[2*k1][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[3][col] = 1; V[p+k0][col] = 1; V[k0][col] = -1; W[2*k0][col] = 1; W[2*k0+1][col] = 1; col += 1
            U[0][col] = 1; U[1][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = -1; W[2*k1][col] = 1; col += 1
            U[2][col] = 1; U[0][col] = -1; V[k0][col] = 1; V[k1][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[1][col] = 1; U[3][col] = -1; V[p+k0][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = 1; col += 1

        def add_schoolbook(k):
            nonlocal col
            U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
            U[1][col] = 1; V[p+k][col] = 1; W[2*k][col] = 1; col += 1
            U[2][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
            U[3][col] = 1; V[p+k][col] = 1; W[2*k+1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_schoolbook(8)
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    U32, V32, W32 = build_rank32()

    # Verify rank-32
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U32, V32, W32):
        return trivial_decomposition()

    # Try to find rank-31 by merging one schoolbook term into a Strassen block
    # Strategy: take the last 7 terms of Strassen(6,7) + 4 terms of schoolbook(8) = 11 terms
    # Try to reduce to 10 terms via optimization
    
    # Initialize rank-31: take first 32-1=31 terms from rank-32, but we need to merge
    # Better approach: initialize with rank-32, zero out one column, and optimize
    
    def try_rank31():
        if time.time() > deadline - 5:
            return None
        
        # Start from rank-32, remove one column (try removing column 31, the last schoolbook term)
        # and optimize the remaining 31 columns to compensate
        rank = 31
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Copy first 31 columns from rank-32
        U[:, :31] = U32[:, :31]
        V[:, :31] = V32[:, :31]
        W[:, :31] = W32[:, :31]
        
        # The removed term was: U[3]=1, V[17]=1, W[17]=1 (A[1,1]*B[1,8] -> C[1,8])
        # We need to recover this contribution through the remaining terms
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e10
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_flat = params[2*n*m*rank:].reshape(p*n, rank)
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        residual += (val - 1.0)**2
            return residual
        
        def gradient(params):
            if time.time() > deadline - 5:
                return np.zeros_like(params)
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_flat = params[2*n*m*rank:].reshape(p*n, rank)
            
            dU = np.zeros_like(U_flat)
            dV = np.zeros_like(V_flat)
            dW = np.zeros_like(W_flat)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        err = 2 * (val - 1.0)
                        for c in range(rank):
                            dU[i*m+j, c] += err * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                            dV[j*p+k, c] += err * U_flat[i*m+j, c] * W_flat[k*n+i, c]
                            dW[k*n+i, c] += err * U_flat[i*m+j, c] * V_flat[j*p+k, c]
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-10:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_opt = result.x[2*n*m*rank:].reshape(p*n, rank)
                
                if verify(U_opt, V_opt, W_opt):
                    # Convert to lists with integer/half-integer entries if possible
                    U_list = [[float(U_opt[i][j]) for j in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_opt[i][j]) for j in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_opt[i][j]) for j in range(rank)] for i in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
        
        return None

    result = try_rank31()
    if result is not None:
        return result

    # Return rank-32
    U_list = [[float(U32[i][j]) for j in range(32)] for i in range(n*m)]
    V_list = [[float(V32[i][j]) for j in range(32)] for i in range(m*p)]
    W_list = [[float(W32[i][j]) for j in range(32)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
