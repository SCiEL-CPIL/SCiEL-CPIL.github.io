import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T flattened
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We need to find U, V, W such that T = sum U_r V_r W_r
    
    # We search for a decomposition of rank R
    # We iterate R from lower bound to 30
    
    # Lower bound for R(2,3,5) is 15
    for R in range(15, 31):
        # Try to find a decomposition of rank R
        # We fix U and V to have small integer entries
        # and solve for W
        
        # We use a heuristic: pick U, V randomly from {-1, 0, 1}
        # and check if W has small rational entries
        
        found = False
        for trial in range(1000):
            # Generate random U, V with entries in {-1, 0, 1}
            # We want them to be "generic" enough to span the space
            U = np.random.choice([-1, 0, 1], size=(nm, R))
            V = np.random.choice([-1, 0, 1], size=(mp, R))
            
            # Construct B matrix of size nm x R
            # B[:, r] = vec(u_r v_r^T)
            # u_r is column r of U (size nm)
            # v_r is column r of V (size mp)
            # Wait, u_r is in F^n, v_r is in F^m?
            # U is (n*m, R). So u_r is a vector of length n*m.
            # This is not a rank-1 matrix in F^n x F^m.
            # The tensor decomposition is U (nm x R), V (mp x R), W (pn x R).
            # The term is u_r (nm) * v_r (mp) * w_r (pn).
            # This is not a standard matrix multiplication tensor decomposition
            # unless we interpret the indices correctly.
            
            # The tensor T is in F^{nm} x F^{mp} x F^{pn}.
            # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
            # We want T = sum_r u_r v_r^T w_r^T?
            # No, T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
            
            # So for each c, T[:, :, c] = sum_r W[c, r] (U[:, r] v_r^T?)
            # No, T[:, :, c] is a matrix of size nm x mp.
            # T[:, :, c] = sum_r W[c, r] (U[:, r] * V[:, r]^T)
            # Here U[:, r] is a vector of length nm.
            # V[:, r] is a vector of length mp.
            # So U[:, r] * V[:, r]^T is a matrix of size nm x mp.
            
            # So we can form a matrix B of size nm*mp x R
            # where column r is vec(U[:, r] * V[:, r]^T)
            
            # Then for each c, vec(T[:, :, c]) = B * W[c, :]
            
            # We can solve for W if B has full column rank.
            
            B = np.zeros((nm * mp, R))
            for r in range(R):
                # Outer product of U[:, r] and V[:, r]
                outer = np.outer(U[:, r], V[:, r])
                B[:, r] = outer.flatten()
            
            # Check if B has full column rank
            if np.linalg.matrix_rank(B) < R:
                continue
                
            # Solve for W
            W = np.zeros((pn, R))
            valid = True
            for c in range(pn):
                # Get target vector for this c
                # T[:, :, c] is zero everywhere except specific indices
                # We need to construct T[:, :, c]
                
                # Instead of constructing T, we know T is sparse.
                # But we can just construct the full T once.
                
                pass # Placeholder
            
            # Construct T once outside the loop?
            # T is fixed.
            
            # Let's construct T_flat for each c
            # T_c is a matrix of size nm x mp
            # T_c[a, b] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
            
            # We can precompute T_slices
            T_slices = []
            for c in range(pn):
                T_c = np.zeros((nm, mp))
                # Find i, j, k such that c = k*n + i
                k = c // n
                i = c % n
                # b = j*p + k
                # a = i*m + j
                # j can be 0..m-1
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    T_c[a, b] = 1
                T_slices.append(T_c.flatten())
            
            # Now solve B * W[c, :] = T_slices[c]
            for c in range(pn):
                # Least squares solution
                # Since B is nm*mp x R, and R <= nm*mp
                # We use np.linalg.lstsq
                W_c, residuals, rank, sv = np.linalg.lstsq(B, T_slices[c], rcond=None)
                W[c, :] = W_c
                
                # Check if solution is exact
                if not np.allclose(B @ W[c, :], T_slices[c]):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if U, V, W have integer/half-integer entries
            # We can scale them?
            # The problem requires the entries in the returned dict to be int/half-int.
            # If U, V, W are rational, we can check if they are close to half-integers.
            
            # Check U
            U_check = np.round(U * 2) / 2
            if not np.allclose(U, U_check):
                continue
            V_check = np.round(V * 2) / 2
            if not np.allclose(V, V_check):
                continue
            W_check = np.round(W * 2) / 2
            if not np.allclose(W, W_check):
                continue
            
            # If we get here, we have a valid decomposition
            return {
                "U": U_check.tolist(),
                "V": V_check.tolist(),
                "W": W_check.tolist()
            }
            
        # If no solution found for this R, try next R

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
# EVOLVE-BLOCK-END
