import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # Dimensions of U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Precompute the target tensor entries
    # T[(i,j), (j',k), (k',i')] = 1 if i==i', j==j', k==k' else 0
    # We will compute the error directly in the loop for efficiency
    
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from low to high
    # Start from 1, go up to 36
    for rank in range(1, n * m * p + 1):
        found = False
        # Try multiple random initializations
        for trial in range(20):
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(dim_U, rank) * 0.5
            V = np.random.randn(dim_V, rank) * 0.5
            W = np.random.randn(dim_W, rank) * 0.5
            
            # Simple Gradient Descent
            # Error E = sum (sum_s U_... V_... W_... - T_...)^2
            # We can use a simple iterative update or scipy
            # Given time limit, let's try a few iterations of ALS with Zero Constraints?
            # ALS with zero constraints is complex.
            # Let's try to use the fact that we can check validity exactly.
            
            # Alternative: Use the current ALS but add penalty for off-diagonals?
            # Or just use scipy.optimize.minimize on the flattened vector?
            # Flattened size: (9+12+12)*rank = 33*rank.
            # For rank=10, size=330. Manageable.
            
            def error_func(params):
                U_f = params[:dim_U*rank].reshape(dim_U, rank)
                V_f = params[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
                W_f = params[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
                
                err = 0.0
                # Iterate over all tensor entries
                # i, j, j', k, k', i'
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        # Target
                                        target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        # Value
                                        val = 0.0
                                        for s in range(rank):
                                            val += U_f[i*m+j, s] * V_f[jp*p+k, s] * W_f[kp*n+ip, s]
                                        err += (val - target) ** 2
                return err
            
            # Use Nelder-Mead or similar for small problems?
            # Or just do a few steps of gradient descent manually?
            # Let's try to converge using a simple loop with gradient estimation?
            # Too slow.
            
            # Let's stick to the current ALS but ADD the zero constraints to the least squares?
            # When updating W, we have equations for diagonal (target 1) and off-diagonal (target 0).
            # This is a linear system!
            
            # Let's implement ALS with ALL constraints.
            converged = False
            for it in range(100):
                # Update W
                # Fix U, V.
                # For each (i, k) in W, we have variables W[k*n+i, s] for s in 0..rank-1.
                # Constraints:
                # 1. Diagonal: For all j, sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s] = 1
                # 2. Off-diagonal: For all j' != j or k' != k or i' != i?
                #    The term involving W[k*n+i, s] appears in T[(i,j), (j',k'), (k'',i'')]
                #    if the W index matches (k'', i'') = (k, i).
                #    So for fixed W row (k, i), the terms are:
                #    Sum_s U[i*m+j, s] V[j*p+k', s] W[k*n+i, s] ... wait.
                #    The tensor entry is T[(i,j), (j',k'), (k'',i'')].
                #    It involves U[i*m+j], V[j'*p+k'], W[k''*n+i''].
                #    If we fix W[k*n+i], then k''=k, i''=i.
                #    The equation is:
                #    Sum_s U[i*m+j, s] V[j'*p+k', s] W[k*n+i, s] = Target(i,j, j',k', k,i)
                #    Target is 1 if i==i' (always i), j==j', k==k'.
                #    So Target is 1 if j==j' and k==k'. Else 0.
                
                # So for fixed W row (k,i), we have a linear system for W[k*n+i, :].
                # Variables: w_s = W[k*n+i, s].
                # Equations for each (j, j', k'):
                # Sum_s w_s * (U[i*m+j, s] * V[j'*p+k', s]) = 1 if (j==j' and k==k') else 0.
                
                A = np.zeros((m * m * p, rank))
                b = np.zeros(m * m * p)
                idx = 0
                for j in range(m):
                    for jp in range(m):
                        for kp in range(p):
                            for s in range(rank):
                                A[idx, s] = U[i*m+j, s] * V[jp*p+kp, s]
                            if j == jp and k == kp:
                                b[idx] = 1.0
                            else:
                                b[idx] = 0.0
                            idx += 1
                
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = sol
                
                # Update U
                # Fix V, W.
                # For each (i, j) in U.
                # Variables u_s = U[i*m+j, s].
                # Equations for each (j', k, k', i'):
                # Sum_s u_s * (V[j'*p+k, s] * W[k'*n+i', s]) = Target(i,j, j',k, k',i')
                # Target is 1 if i==i' (always i), j==j', k==k'.
                # So Target is 1 if j==j' and k==k'. Else 0.
                
                A = np.zeros((m * p * n, rank))
                b = np.zeros(m * p * n)
                idx = 0
                for jp in range(m):
                    for kp in range(p):
                        for ip in range(n):
                            for s in range(rank):
                                A[idx, s] = V[jp*p+kp, s] * W[kp*n+ip, s]
                            if j == jp and k == kp:
                                b[idx] = 1.0
                            else:
                                b[idx] = 0.0
                            idx += 1
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = sol
                
                # Update V
                # Fix U, W.
                # For each (j, k) in V.
                # Variables v_s = V[j*p+k, s].
                # Equations for each (i, j', k', i'):
                # Sum_s v_s * (U[i*m+j', s] * W[k'*n+i', s]) = Target(i,j', j,k, k',i')
                # Target is 1 if i==i', j'==j, k==k'.
                # So Target is 1 if i==i' and j'==j and k==k'.
                # Note: The V index is (j,k). The equation involves V[j*p+k].
                # So we fix j,k.
                # The term is U[i*m+j', s] * V[j*p+k, s] * W[k'*n+i', s].
                # Wait, the V index in the tensor is (j',k')?
                # Tensor T[(i,j), (j',k'), (k'',i'')].
                # If we fix V row (j,k), then in the sum, the V term is V[j'*p+k'].
                # So we need j'=j and k'=k.
                # So the equation is for fixed j,k:
                # Sum_s v_s * (U[i*m+j', s] * W[k'*n+i', s]) ... wait.
                # The V term is V[j'*p+k']. If we are updating V[j*p+k], then j'=j, k'=k.
                # The other terms are U[i*m+j', s] and W[k'*n+i', s].
                # Wait, the U index is (i,j). The V index is (j',k'). The W index is (k'',i'').
                # In the product U[i*m+j] V[j'*p+k'] W[k''*n+i''], the j in U and j' in V are DIFFERENT variables in the tensor definition?
                # No, in the matrix multiplication tensor, the indices are coupled.
                # T_{(i,j), (j',k), (k',i')}?
                # Let's stick to the code's indexing which worked for the diagonal.
                # Code: U[i*m+j], V[j*p+k], W[k*n+i].
                # This implies the tensor is T_{(i,j), (j,k), (k,i)}.
                # But we established the tensor has zeros elsewhere.
                # So T_{(i,j), (j',k'), (k'',i'')} = 1 if i=i', j=j', k=k' else 0.
                
                # When updating V[j*p+k], we fix j,k.
                # The term in the sum is V[j'*p+k'].
                # For this to be the variable we are updating, we must have j'=j and k'=k.
                # So the equation involves U[i*m+j', s] where j' is free?
                # No, the U index is (i,j). The V index is (j',k').
                # In the tensor T, the indices are (i,j), (j',k'), (k'',i'').
                # If we fix V to (j,k), then j'=j, k'=k.
                # The other factors are U[i*m+j', s] and W[k''*n+i', s].
                # Wait, the U index second component is j. The V index first component is j'.
                # In the tensor, are they independent?
                # Yes, the tensor is a 3-way array.
                # The value is 1 only if i=i', j=j', k=k'.
                # So for fixed V(j,k), we sum over s.
                # The term is U[i*m+j', s] * V[j*p+k, s] * W[k''*n+i', s].
                # Here j' is the index of U, k'' is the index of W.
                # Wait, U is indexed by (i,j). V by (j',k'). W by (k'',i'').
                # If we fix V to (j,k), then j'=j, k'=k.
                # The equation is:
                # Sum_s v_s * (U[i*m+j', s] * W[k''*n+i', s]) = Target(i,j', j,k, k'',i')
                # Target is 1 if i=i', j'=j, k=k''.
                # So Target is 1 if i=i' and j'=j and k''=k.
                
                A = np.zeros((n * m * n * p, rank)) # i, j', k'', i'
                b = np.zeros(n * m * n * p)
                idx = 0
                for i in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                for s in range(rank):
                                    A[idx, s] = U[i*m+jp, s] * W[kp*n+ip, s]
                                if i == ip and jp == j and kp == k:
                                    b[idx] = 1.0
                                else:
                                    b[idx] = 0.0
                                idx += 1
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = sol
                
                # Check residual
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        val = np.dot(U[i*m+j, :], V[jp*p+kp, :] * W[kp*n+ip, :])
                                        err = abs(val - target)
                                        if err > max_err:
                                            max_err = err
                if max_err < 1e-6:
                    converged = True
                    break
            
            if converged:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                valid = True
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        val = np.dot(U_r[i*m+j, :], V_r[jp*p+kp, :] * W_r[kp*n+ip, :])
                                        if abs(val - target) > 1e-9:
                                            valid = False
                                            break
                                    if not valid: break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if valid:
                    best_rank = rank
                    best_solution = (U_r, V_r, W_r)
                    found = True
                    break
        if found:
            break

    if best_solution is not None:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
