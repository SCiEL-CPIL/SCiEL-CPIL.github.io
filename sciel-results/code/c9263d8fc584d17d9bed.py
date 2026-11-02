"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # T[i,j,k] = 1 if i == k else 0
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == k:
                    T[i, j, k] = 1.0
    
    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # A[i*m+j, r] = U row
        # B[j*p+k, r] = V row  
        # C[k*n+i, r] = W row
        # Constraint: sum_r A[i*m+j,r] * B[j*p+k,r] * C[k*n+i,r] = T[i,j,k]
        
        A = np.random.randn(n * m, r) * 0.1
        B = np.random.randn(m * p, r) * 0.1
        C = np.random.randn(p * n, r) * 0.1
        
        for iteration in range(300):
            if time.time() > deadline:
                return None
            
            # Update C (W): for each (k,i), the constraint is
            # sum_r C[k*n+i,r] * (sum_j A[i*m+j,r] * B[j*p+k,r]) = sum_j T[i,j,k]
            # Let S[k,i] = sum_j T[i,j,k]
            # Let M[k,i,r] = sum_j A[i*m+j,r] * B[j*p+k,r]
            # Then: C[k*n+i,:] dot M[k,i,:] = S[k,i]
            
            S = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    S[k, i] = m if i == k else 0.0
            
            M = np.zeros((p, n, r))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        M[k, i, :] += A[i*m+j, :] * B[j*p+k, :]
            
            for k in range(p):
                for i in range(n):
                    mvec = M[k, i, :]
                    mnorm2 = np.dot(mvec, mvec)
                    if mnorm2 > 1e-14:
                        C[k*n+i, :] = S[k, i] * mvec / mnorm2
                    else:
                        C[k*n+i, :] = 0.0
            
            # Update A (U): for each (i,j), constraints over k:
            # For each k: sum_r A[i*m+j,r] * B[j*p+k,r] * C[k*n+i,r] = T[i,j,k]
            # Let Q[i,j,k,r] = B[j*p+k,r] * C[k*n+i,r]
            # Then: A[i*m+j,:] dot Q[i,j,k,:] = T[i,j,k] for all k
            # Stack: A[i*m+j,:] @ Q[i,j,:].T = T[i,j,:]
            
            Q = np.zeros((n, m, p, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Q[i, j, k, :] = B[j*p+k, :] * C[k*n+i, :]
            
            for i in range(n):
                for j in range(m):
                    Qt = Q[i, j, :, :].T  # (r, p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(Qt, T[i, j, :], rcond=None)
                        A[i*m+j, :] = sol
                    except:
                        pass
            
            # Update B (V): for each (j,k), constraints over i:
            # For each i: sum_r B[j*p+k,r] * A[i*m+j,r] * C[k*n+i,r] = T[i,j,k]
            # Let R[j,k,i,r] = A[i*m+j,r] * C[k*n+i,r]
            # Then: B[j*p+k,:] @ R[j,k,:].T = T[:,j,k]
            
            R = np.zeros((m, p, n, r))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        R[j, k, i, :] = A[i*m+j, :] * C[k*n+i, :]
            
            for j in range(m):
                for k in range(p):
                    Rt = R[j, k, :, :].T  # (r, n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(Rt, T[:, j, k], rcond=None)
                        B[j*p+k, :] = sol
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(A[i*m+j, :], B[j*p+k, :] * C[k*n+i, :])
                        err += (val - T[i, j, k]) ** 2
            
            if err < 1e-12:
                U_list = [[float(x) for x in row] for row in A]
                V_list = [[float(x) for x in row] for row in B]
                W_list = [[float(x) for x in row] for row in C]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 21 upward
    for r in range(21, best_rank + 1):
        if time.time() > deadline:
            break
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result is not None:
                best_result = result
                best_rank = r
                break
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
