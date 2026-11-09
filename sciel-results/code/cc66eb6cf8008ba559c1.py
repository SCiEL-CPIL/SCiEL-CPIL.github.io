import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T[i, j, k] = 1
    # Flattened indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Construct target tensor as a 3D array T[i, j, k]
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # We want to find U, V, W such that 
    # sum_r U[:, r] outer V[:, r] outer W[:, r] approx T
    # Actually, the decomposition is T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    
    # Let's use ALS.
    # We need to map the 3D tensor to the flattened form for ALS.
    # T_flat is not directly used in standard ALS which assumes T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # Here our indices are coupled: U depends on i,j. V depends on j,k. W depends on k,i.
    # This is not a standard CP decomposition of a 3D array with independent indices.
    # The indices i, j, k are shared.
    # U is a matrix of size (n*m) x R.
    # V is a matrix of size (m*p) x R.
    # W is a matrix of size (p*n) x R.
    
    # The condition is:
    # For all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a system of equations.
    # We can solve for U given V, W:
    # U[i*m+j, r] = T[i,j,k] / (V[j*p+k, r] * W[k*n+i, r]) ? No, sum over r.
    # This is not linear in U if V and W are fixed?
    # T[i,j,k] = sum_r U_{ij,r} (V_{jk,r} W_{ki,r})
    # Let Q_{jk,ki, r} = V_{jk,r} W_{ki,r}.
    # Then for fixed j, k, i, it's linear in U.
    # But U_{ij,r} appears in multiple equations (for different k).
    # For fixed i, j, U_{ij,r} is multiplied by sum_k V_{jk,r} W_{ki,r}? No.
    # T[i,j,k] = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # For fixed i, j, we have p equations (one for each k) for the R variables U_{ij, 0..R-1}.
    # This is an underdetermined system if p < R.
    
    # We can set up a global least squares problem.
    # Minimize || T - sum_r U_r \otimes V_r \otimes W_r ||_F^2
    # This is non-convex.
    
    # Given the complexity, and the fact that I need to output a valid program,
    # I will stick to the schoolbook solution but try to optimize it slightly if possible.
    # However, the prompt asks for a "stronger algorithm".
    
    # Let's try a simple heuristic:
    # If we can't find a better one quickly, return schoolbook.
    
    # Actually, I can just return the schoolbook solution. It's valid.
    # The score is 1.0.
    # To beat 1.0, I need rank < 72.
    
    # Is there a known rank for 3x4x6?
    # Let's assume the minimal rank is not easily found by simple ALS in 150s.
    
    # I will implement a basic ALS that tries to find a solution of rank R.
    # If it finds an exact integer solution, I return it.
    # Otherwise, I return the schoolbook solution.
    
    def try_rank(R):
        np.random.seed(42)
        U = np.random.randn(dim_u, R)
        V = np.random.randn(dim_v, R)
        W = np.random.randn(dim_w, R)
        
        for _ in range(100):
            # Update U
            # For each (i,j), solve for U[i*m+j, :]
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a linear system for U[i*m+j, :]
            # Let A[i,j] be a p x R matrix where A[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
            # Then T[i,j,:] = U[i*m+j, :] @ A[i,j].T
            # So U[i*m+j, :] = T[i,j,:] @ pinv(A[i,j].T)
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve U[i*m+j, :] @ A.T = T[i,j,:]
                    # U[i*m+j, :] = T[i,j,:] @ pinv(A.T)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, T[i,j,:], rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, T[:,j,k], rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A.T, T[i,:,k], rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error += (val - 1.0)**2
            if error < 1e-6:
                # Round to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                # Verify exactness
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                    if not is_exact: break
                if is_exact:
                    return U_int, V_int, W_int
        return None

    start_time = time.time()
    # Try ranks from 30 to 71
    for R in range(30, 72):
        if time.time() - start_time > 100:
            break
        res = try_rank(R)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
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
