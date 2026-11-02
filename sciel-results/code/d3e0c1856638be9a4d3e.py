import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] = 1
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # But tensor rank is typically higher. Let's try from 30 up.
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    # Schoolbook solution as fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    # Use random initialization and gradient descent / least squares
    np.random.seed(42)
    
    for rank in range(20, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Use alternating least squares
            # For fixed V,W: solve for U
            # For fixed U,W: solve for V  
            # For fixed U,V: solve for W
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each (i,j), sum_alpha U[i*m+j,alpha] * V[j*p+k,alpha] * W[k*n+i,alpha] = 1
                # This is: sum_k (V[j*p+k,:] * W[k*n+i,:]) dot U[i*m+j,:] = 1
                # Let M_U[i*m+j, alpha] = sum_k V[j*p+k, alpha] * W[k*n+i, alpha]
                # Then U[i*m+j,:] * M_U[i*m+j,:] = 1... no, that's not right.
                
                # Actually: sum_alpha U[i*m+j, alpha] * (sum_k V[j*p+k, alpha] * W[k*n+i, alpha]) = 1
                # Let S_U[i*m+j, alpha] = sum_k V[j*p+k, alpha] * W[k*n+i, alpha]
                # Then we need: U[i*m+j, :] dot S_U[i*m+j, :] = 1
                # This is a system: for each (i,j), U[i*m+j,:] is a row vector, and we need
                # U[i*m+j,:] @ S_U[i*m+j,:] = 1
                # But S_U depends on V and W, which are fixed during U update.
                # This gives one equation per (i,j) with rank unknowns. Underdetermined if rank > 1.
                # We can minimize ||U[i*m+j,:] @ S_U[i*m+j,:] - 1||^2
                # The minimum norm solution: U[i*m+j,:] = S_U[i*m+j,:] / (S_U[i*m+j,:] @ S_U[i*m+j,:]) if norm > 0
                
                # Compute S_U
                S_U = np.zeros((nm, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S_U[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
                
                # Solve for U: minimize sum_{i,j} (U[i*m+j,:] @ S_U[i*m+j,:] - 1)^2
                # Gradient: 2 * U[i*m+j,:] * S_U[i*m+j,:] * (U[i*m+j,:] @ S_U[i*m+j,:] - 1)
                # Set to 0: U[i*m+j,:] * S_U[i*m+j,:] * (U[i*m+j,:] @ S_U[i*m+j,:] - 1) = 0
                # Solution: U[i*m+j,:] = c * S_U[i*m+j,:] where c * S_U[i*m+j,:] @ S_U[i*m+j,:] = 1
                # So c = 1 / (S_U[i*m+j,:] @ S_U[i*m+j,:])
                
                for idx in range(nm):
                    s = S_U[idx, :]
                    norm_sq = s @ s
                    if norm_sq > 1e-12:
                        U[idx, :] = s / norm_sq
                    else:
                        # Fallback: random
                        U[idx, :] = np.random.randn(rank)
                
                # Update V similarly
                S_V = np.zeros((mp, rank))
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            S_V[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
                
                for idx in range(mp):
                    s = S_V[idx, :]
                    norm_sq = s @ s
                    if norm_sq > 1e-12:
                        V[idx, :] = s / norm_sq
                    else:
                        V[idx, :] = np.random.randn(rank)
                
                # Update W similarly
                S_W = np.zeros((pn, rank))
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            S_W[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
                
                for idx in range(pn):
                    s = S_W[idx, :]
                    norm_sq = s @ s
                    if norm_sq > 1e-12:
                        W[idx, :] = s / norm_sq
                    else:
                        W[idx, :] = np.random.randn(rank)
                
                # Check residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-10:
                    converged = True
                    break
            
            if converged:
                found = True
                break
        
        if found:
            best_rank = rank
            best_result = (U, V, W)
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # For now, round to nearest 0.5
    U_list = [[round(x * 2) / 2 for x in row] for row in best_result[0]]
    V_list = [[round(x * 2) / 2 for x in row] for row in best_result[1]]
    W_list = [[round(x * 2) / 2 for x in row] for row in best_result[2]]
    
    # Verify the decomposition
    U_arr = np.array(U_list)
    V_arr = np.array(V_list)
    W_arr = np.array(W_list)
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                residual += (val - 1.0) ** 2
    
    if residual > 1e-8:
        # Use schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
