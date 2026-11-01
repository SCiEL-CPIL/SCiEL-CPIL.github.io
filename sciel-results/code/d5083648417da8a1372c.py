import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
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
    
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            
            U[2][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1
            W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
            col += 1
            
            U[0][col] = 1
            V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
            W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            
            U[3][col] = 1
            V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
            W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
            col += 1
            
            U[0][col] = 1; U[1][col] = 1
            V[1*p + k1][col] = 1
            W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
            col += 1
            
            U[2][col] = 1; U[0][col] = -1
            V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            U[1][col] = 1; U[3][col] = -1
            V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + 0][col] = 1
            col += 1
            
            U[1][col] = 1
            V[1*p + k][col] = 1
            W[k*n + 0][col] = 1
            col += 1
            
            U[2][col] = 1
            V[0*p + k][col] = 1
            W[k*n + 1][col] = 1
            col += 1
            
            U[3][col] = 1
            V[1*p + k][col] = 1
            W[k*n + 1][col] = 1
            col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_trivial(6)
        add_strassen(7, 8)
        
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build rank-32 solution
    U32, V32, W32 = build_rank32()
    
    if time.time() > deadline:
        return trivial_decomposition()
    
    # Try to find rank-31 via local search
    U31, V31, W31 = None, None, None
    
    def try_rank31_search():
        nonlocal U31, V31, W31
        if time.time() > deadline - 5:
            return
        
        rank31 = 31
        U_dim = n * m  # 4
        V_dim = m * p  # 18
        W_dim = p * n  # 18
        
        # Start from rank-32, set last column to zero
        U0 = np.array(U32, dtype=float)[:, :rank31].copy()
        V0 = np.array(V32, dtype=float)[:, :rank31].copy()
        W0 = np.array(W32, dtype=float)[:, :rank31].copy()
        
        # Add small random perturbation
        np.random.seed(42)
        U0 += 0.01 * np.random.randn(U_dim, rank31)
        V0 += 0.01 * np.random.randn(V_dim, rank31)
        W0 += 0.01 * np.random.randn(W_dim, rank31)
        
        # Define smooth objective: sum of squared residuals + penalty for large values
        def objective(params):
            U_p = params[:U_dim*rank31].reshape(U_dim, rank31)
            V_p = params[U_dim*rank31:(U_dim+V_dim)*rank31].reshape(V_dim, rank31)
            W_p = params[(U_dim+V_dim)*rank31:].reshape(W_dim, rank31)
            
            # Compute tensor entries
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_p[i*m+j, :] * V_p[j*p+k, :] * W_p[k*n+i, :])
                        residual += (val - 1.0)**2
            
            # Penalty for large entries (encourages sparsity-like structure)
            penalty = 0.01 * (np.sum(U_p**2) + np.sum(V_p**2) + np.sum(W_p**2))
            
            return residual + penalty
        
        def gradient(params):
            U_p = params[:U_dim*rank31].reshape(U_dim, rank31)
            V_p = params[U_dim*rank31:(U_dim+V_dim)*rank31].reshape(V_dim, rank31)
            W_p = params[(U_dim+V_dim)*rank31:].reshape(W_dim, rank31)
            
            gU = np.zeros_like(U_p)
            gV = np.zeros_like(V_p)
            gW = np.zeros_like(W_p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_p[i*m+j, :] * V_p[j*p+k, :] * W_p[k*n+i, :])
                        err = 2 * (val - 1.0)
                        
                        for c in range(rank31):
                            gU[i*m+j, c] += err * V_p[j*p+k, c] * W_p[k*n+i, c]
                            gV[j*p+k, c] += err * U_p[i*m+j, c] * W_p[k*n+i, c]
                            gW[k*n+i, c] += err * U_p[i*m+j, c] * V_p[j*p+k, c]
            
            # Add penalty gradient
            gU += 0.02 * U_p
            gV += 0.02 * V_p
            gW += 0.02 * W_p
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        # Gradient descent with line search
        params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        lr = 0.01
        max_iter = 200
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            grad = gradient(params)
            params -= lr * grad
            
            # Periodic check
            if it % 20 == 0:
                U_check = params[:U_dim*rank31].reshape(U_dim, rank31)
                V_check = params[U_dim*rank31:(U_dim+V_dim)*rank31].reshape(V_dim, rank31)
                W_check = params[(U_dim+V_dim)*rank31:].reshape(W_dim, rank31)
                
                if verify(U_check.tolist(), V_check.tolist(), W_check.tolist(), tol=1e-6):
                    U31 = U_check.tolist()
                    V31 = V_check.tolist()
                    W31 = W_check.tolist()
                    return
        
        # Final check
        U_check = params[:U_dim*rank31].reshape(U_dim, rank31)
        V_check = params[U_dim*rank31:(U_dim+V_dim)*rank31].reshape(V_dim, rank31)
        W_check = params[(U_dim+V_dim)*rank31:].reshape(W_dim, rank31)
        
        if verify(U_check.tolist(), V_check.tolist(), W_check.tolist(), tol=1e-6):
            U31 = U_check.tolist()
            V31 = V_check.tolist()
            W31 = W_check.tolist()
    
    try_rank31_search()
    
    # If rank-31 found, return it; otherwise return rank-32
    if U31 is not None:
        # Round to nearest integer/half-integer if close
        def round_matrix(M):
            result = []
            for row in M:
                new_row = []
                for val in row:
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-4:
                        new_row.append(rounded)
                    else:
                        new_row.append(val)
                result.append(new_row)
            return result
        
        U31_r = round_matrix(U31)
        V31_r = round_matrix(V31)
        W31_r = round_matrix(W31)
        
        if verify(U31_r, V31_r, W31_r, tol=1e-8):
            return {"U": U31_r, "V": V31_r, "W": W31_r}
        else:
            return {"U": U31, "V": V31, "W": W31}
    else:
        # Return rank-32
        return {"U": U32, "V": V32, "W": W32}
# EVOLVE-BLOCK-END
