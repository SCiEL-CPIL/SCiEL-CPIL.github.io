import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    rank = n * m * p  # start with schoolbook
    deadline = time.time() + 100  # safety margin
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
    
    # Try to find lower rank decompositions
    best_rank = rank
    best_U = None
    best_V = None
    best_W = None
    
    # Start from schoolbook
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, tol=1e-10):
        n, m, p = 2, 4, 5
        # Check all (a,b,c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1.0) > tol:
                        return False
        # Check some zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (i_a == i_c and j_a == j_b and k_b == k_c):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    # Try ALS from schoolbook initialization with lower rank
    for target_rank in range(39, 15, -1):
        if time.time() > deadline - 20:
            break
        # Initialize by truncating the schoolbook
        U_init = U[:, :target_rank]
        V_init = V[:, :target_rank]
        W_init = W[:, :target_rank]
        
        # Run ALS
        U_als = U_init.copy()
        V_als = V_init.copy()
        W_als = W_init.copy()
        
        converged = False
        for iteration in range(200):
            if time.time() > deadline - 20:
                break
            
            # Update U: for each a, solve sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # This is: for each a, U[a,:] * (V[b,:] * W[c,:])^T = T[a,b,c] for all (b,c)
            # Let M_a[b,c] = V[b,:] * W[c,:] (element-wise product)
            # Then U[a,:] @ M_a^T should give... hmm, this isn't quite right.
            
            # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For fixed a: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            # Let's define for each r: vector v_r = V[:,r], w_r = W[:,r]
            # Then: sum_r U[a,r] * (v_r * w_r)[b] * ... no.
            
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For fixed a: this is a bilinear form.
            # Let's think of it as: U[a,:] is a row vector u_a
            # We need: sum_r u_a[r] * V[b,r] * W[c,r] = T[a,b,c]
            
            # For fixed a, define matrix M_a where M_a[b,c] = sum_r ... no.
            # Actually, for fixed a, the equation is:
            # For all (b,c): sum_r u_a[r] * (V[b,r] * W[c,r]) = T[a,b,c]
            # Let P_r = V[:,r] * W[:,r] (element-wise, shape (m*p, p*n))
            # Then: sum_r u_a[r] * P_r[b,c] = T[a,b,c]
            # So: sum_r u_a[r] * P_r = T[a] (as matrices)
            # This is a linear system in u_a: [P_0, P_1, ..., P_{r-1}] @ u_a = T[a]
            
            # P_r is (m*p) x (p*n), so we have (m*p)*(p*n) equations and r unknowns.
            # This is overdetermined, so we solve least squares.
            
            m_dim = m * p  # 20
            n_dim = p * n  # 10
            r = target_rank
            
            # Build the system for U
            # For each r, P_r = V_als[:,r] * W_als[:,r] (outer product? no, element-wise)
            # Wait, V[b,r] * W[c,r] is element-wise product of vectors V[:,r] and W[:,r]
            # So P_r = V_als[:,r] * W_als[:,r] gives a matrix of shape (m*p, p*n)? No!
            # V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
            # V[b,r] * W[c,r] for all b,c is the outer product V[:,r] @ W[:,r].T
            # But that's not what we want. We want V[b,r] * W[c,r] which is just the product of scalars.
            # So for fixed r, the "matrix" P_r[b,c] = V[b,r] * W[c,r] is indeed the outer product.
            
            # So: sum_r u_a[r] * (V_als[:,r] @ W_als[:,r].T) = T[a]
            # This is: sum_r u_a[r] * outer(V_als[:,r], W_als[:,r]) = T[a]
            
            # Let's vectorize: for each r, vec(outer(V[:,r], W[:,r])) = V[:,r] ⊗ W[:,r] (Kronecker)
            # Then: sum_r u_a[r] * (V[:,r] ⊗ W[:,r]) = vec(T[a])
            # This is a linear system: A @ u_a = vec(T[a]) where A[r] = V[:,r] ⊗ W[:,r]
            
            # A has shape (m*p * p*n, r) = (200, r)
            # vec(T[a]) has shape (200,)
            
            # Build A
            A = np.zeros((m_dim * n_dim, r))
            for rr in range(r):
                A[:, rr] = np.outer(V_als[:, rr], W_als[:, rr]).flatten()
            
            # Solve for each a
            for a in range(n * m):
                rhs = T[a].flatten()
                # Least squares: A @ u_a = rhs
                try:
                    u_a, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    U_als[a, :] = u_a
                except:
                    pass
            
            # Similarly update V and W
            # For V: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For fixed b: sum_r V[b,r] * (U[a,r] * W[c,r]) = T[a,b,c] for all (a,c)
            # A_v[r] = U_als[:,r] ⊗ W_als[:,r] (shape (n*m, p*n))
            # A_v_flat has shape (n*m * p*n, r) = (80, r)
            A_v = np.zeros((n * m * n_dim, r))
            for rr in range(r):
                A_v[:, rr] = np.outer(U_als[:, rr], W_als[:, rr]).flatten()
            
            for b in range(m * p):
                rhs = T[:, b, :].flatten()
                try:
                    v_b, _, _, _ = np.linalg.lstsq(A_v, rhs, rcond=None)
                    V_als[b, :] = v_b
                except:
                    pass
            
            # For W: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For fixed c: sum_r W[c,r] * (U[a,r] * V[b,r]) = T[a,b,c] for all (a,b)
            A_w = np.zeros((n * m * m * p, r))
            for rr in range(r):
                A_w[:, rr] = np.outer(U_als[:, rr], V_als[:, rr]).flatten()
            
            for c in range(p * n):
                rhs = T[:, :, c].flatten()
                try:
                    w_c, _, _, _ = np.linalg.lstsq(A_w, rhs, rcond=None)
                    W_als[c, :] = w_c
                except:
                    pass
        
        # Check if we got a valid solution
        if verify(U_als, V_als, W_als):
            best_rank = target_rank
            best_U = U_als
            best_V = V_als
            best_W = W_als
            break  # Found the lowest rank
    
    # If no lower rank found, return schoolbook
    if best_U is None:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Round to rationals if possible
        def round_to_rationals(mat):
            mat_r = mat.copy()
            for i in range(mat_r.shape[0]):
                for j in range(mat_r.shape[1]):
                    # Try to round to nearest half-integer
                    val = mat_r[i,j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        mat_r[i,j] = rounded
                    else:
                        rounded2 = round(val)
                        if abs(val - rounded2) < 1e-6:
                            mat_r[i,j] = rounded2
            return mat_r
        
        U_out = round_to_rationals(best_U)
        V_out = round_to_rationals(best_V)
        W_out = round_to_rationals(best_W)
        
        # Verify after rounding
        if verify(U_out, V_out, W_out):
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        else:
            # Return unrounded
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
