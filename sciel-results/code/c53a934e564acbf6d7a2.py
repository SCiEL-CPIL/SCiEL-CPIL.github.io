import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Target: T[i,j,k] = 1 for all i,j,k
    # Constraint: sum_c U[nm_idx, c] * V[mp_idx, c] * W[pn_idx, c] = 1
    # where nm_idx = i*m+j, mp_idx = j*p+k, pn_idx = k*n+i
    
    # Build index mapping
    # For each (i,j,k), we have:
    #   U index: i*m + j
    #   V index: j*p + k  
    #   W index: k*n + i
    
    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin
    
    # Try ranks from low to high
    best_result = None
    
    for rank in range(1, 55):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            converged = False
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row a, U[a,:] should satisfy
                # sum_c U[a,c] * V[mp_idx(a,k),c] * W[pn_idx(a,k),c] = 1 for all k
                # This is a linear system in U[a,:]
                # Actually, let's think about this more carefully.
                
                # The constraint is: for all (i,j,k):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                
                # Fix V and W, update U:
                # For each (i,j), we need: for all k,
                # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # This is a linear system: M_{ij} @ U[i*m+j, :] = 1
                # where M_{ij}[k, c] = V[j*p+k, c] * W[k*n+i, c]
                
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        # Build M: shape (p, rank)
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve M @ x = ones(p)
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[a, :] = sol
                        except:
                            pass
                
                # Fix U and W, update V:
                # For each (j,k), we need: for all i,
                # sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c] = 1
                # M_{jk}[i, c] = U[i*m+j, c] * W[k*n+i, c]
                for j in range(m):
                    for k in range(p):
                        b = j * p + k
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[b, :] = sol
                        except:
                            pass
                
                # Fix U and V, update W:
                # For each (k,i), we need: for all j,
                # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
                # M_{ki}[j, c] = U[i*m+j, c] * V[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        c_idx = k * n + i
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[c_idx, :] = sol
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if err < 1e-12:
                    converged = True
                    break
            
            if converged:
                # Try to clean up to integer/half-integer
                # Round to nearest half-integer
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                # Verify the cleaned version
                err_clean = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_clean[i*m+j, :], V_clean[j*p+k, :] * W_clean[k*n+i, :])
                            err_clean += (val - 1.0)**2
                
                if err_clean < 1e-9:
                    # Valid solution with half-integers
                    found = True
                    best_result = (U_clean, V_clean, W_clean, rank)
                    break
                else:
                    # Try with just rounding to integers
                    U_clean = np.round(U)
                    V_clean = np.round(V)
                    W_clean = np.round(W)
                    
                    err_clean = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_clean[i*m+j, :], V_clean[j*p+k, :] * W_clean[k*n+i, :])
                                err_clean += (val - 1.0)**2
                    
                    if err_clean < 1e-9:
                        found = True
                        best_result = (U_clean, V_clean, W_clean, rank)
                        break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        U_clean, V_clean, W_clean, rank = best_result
        # Convert to lists of lists
        U_list = [[float(x) for x in row] for row in U_clean]
        V_list = [[float(x) for x in row] for row in V_clean]
        W_list = [[float(x) for x in row] for row in W_clean]
        return {"U": U_list, "V": V_list, "W": W_list}
