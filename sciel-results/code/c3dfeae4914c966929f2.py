import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Target tensor definition
    # T[i, j, k] corresponds to the coefficient of A[i, j] B[j, k] in C[i, k]
    # We need sum_r W[k, i, r] U[i, j, r] V[j, k, r] = delta[i, i'] delta[j, j'] delta[k, k'] ?
    # No, the condition is:
    # sum_r W[k, i, r] U[a, b, r] V[c, d, r] = sum_j delta[a, i] delta[b, j] delta[c, j] delta[d, k]
    
    # Reshape U, V, W to 3D for easier indexing
    # U: (n, m, R)
    # V: (m, p, R)
    # W: (p, n, R)
    
    def error_function(params, R):
        U = params[:n*m*R].reshape(n, m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
        W = params[n*m*R + m*p*R:].reshape(p, n, R)
        
        # Compute the tensor product sum
        # Result[i, k, a, b, c, d] = sum_r W[k, i, r] U[a, b, r] V[c, d, r]
        # We want this to equal Target[i, k, a, b, c, d]
        
        # Target is non-zero only if a==i, d==k, b==c.
        # So we can compute the error more efficiently.
        
        # For each i, k:
        # sum_r W[k, i, r] U[i, :, r] V[:, k, r] should be Identity (m x m)
        # sum_r W[k, i, r] U[a, :, r] V[:, d, r] should be 0 if a!=i or d!=k
        
        err = 0.0
        # Check the diagonal terms (a=i, d=k)
        for i in range(n):
            for k in range(p):
                # Matrix M_jj' = sum_r W[k, i, r] U[i, j, r] V[j', k, r]
                # We want M to be I (identity)
                M = np.zeros((m, m))
                for r in range(R):
                    M += W[k, i, r] * np.outer(U[i, :, r], V[:, k, r])
                err += np.sum((M - np.eye(m))**2)
                
                # Check off-diagonal terms
                # For a != i or d != k
                # sum_r W[k, i, r] U[a, b, r] V[c, d, r] = 0
                # This is equivalent to checking that the vectors are orthogonal in some sense?
                # Actually, we can just sum the squares of all other terms.
                # There are n*m*p*p*n*m - n*p*m terms?
                # Total terms: n * p * n * m * m * p = n^2 m^2 p^2
                # Non-zero terms: n * p * m (for each i, k, one b=c=j)
                # So we check the rest.
                
                # Efficient way:
                # The full tensor F[a, b, c, d] = sum_r W[k, i, r] U[a, b, r] V[c, d, r]
                # We want F[a, b, c, d] = 0 unless a=i, d=k, b=c.
                
                # Let's compute F for all a, d
                # F is (n, m, m, p)
                # F[a, b, c, d] = sum_r W[k, i, r] U[a, b, r] V[c, d, r]
                
                # We can compute this using einsum or loops.
                # Given small sizes, loops are fine.
                F = np.zeros((n, m, m, p))
                for r in range(R):
                    # W[k, i, r] is scalar
                    # U[a, b, r] is (n, m)
                    # V[c, d, r] is (m, p)
                    # F[a, b, c, d] += W * U[a, b] * V[c, d]
                    # This is an outer product of U[:, :, r] and V[:, :, r]
                    # U_slice is (n, m), V_slice is (m, p)
                    # Result is (n, m, m, p)
                    F += W[k, i, r] * np.einsum('ab,cd->abcd', U[:, :, r], V[:, :, r])
                
                # Now check F
                # Target for this i, k:
                # T[a, b, c, d] = 1 if a==i and d==k and b==c else 0
                T = np.zeros((n, m, m, p))
                for b in range(m):
                    T[i, b, b, k] = 1.0
                err += np.sum((F - T)**2)
        
        return err

    # Try ranks from 22 to 24
    for R in range(22, 25):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize
            # U: (n, m, R), V: (m, p, R), W: (p, n, R)
            # Random initialization
            U = np.random.randn(n, m, R)
            V = np.random.randn(m, p, R)
            W = np.random.randn(p, n, R)
            
            # Flatten
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use scipy minimize
            try:
                from scipy.optimize import minimize
                res = minimize(error_function, params, args=(R,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
                if res.success and res.fun < 1e-6:
                    # Extract and round
                    U_opt = res.x[:n*m*R].reshape(n, m, R)
                    V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                    W_opt = res.x[n*m*R + m*p*R:].reshape(p, n, R)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify exactly
                    # Check if the rounded solution is valid
                    valid = True
                    for i in range(n):
                        for k in range(p):
                            F = np.zeros((n, m, m, p))
                            for r in range(R):
                                F += W_rounded[k, i, r] * np.einsum('ab,cd->abcd', U_rounded[:, :, r], V_rounded[:, :, r])
                            T = np.zeros((n, m, m, p))
                            for b in range(m):
                                T[i, b, b, k] = 1.0
                            if not np.allclose(F, T, atol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Convert to lists
                        U_list = U_rounded.flatten().tolist()
                        V_list = V_rounded.flatten().tolist()
                        W_list = W_rounded.flatten().tolist()
                        
                        # Reshape to (rows, rank) as required
                        U_mat = np.array(U_list).reshape(n*m, R)
                        V_mat = np.array(V_list).reshape(m*p, R)
                        W_mat = np.array(W_list).reshape(p*n, R)
                        
                        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            except Exception:
                pass

    return get_schoolbook()
