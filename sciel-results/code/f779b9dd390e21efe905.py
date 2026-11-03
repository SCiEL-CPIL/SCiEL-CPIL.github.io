import time
import numpy as np
import random

DIMENSIONS = [4, 5, 5]

def _verify_strong(U, V, W, n, m, p):
    """
    Verify exact matrix multiplication decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # Check diagonal: sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    # Check off-diagonal: sum_r U[i,j,r] V[j',k,r] W[k,i,r] == 0 for j != j'
    
    # We can compute the full tensor product
    # C[i, k, j, j'] = sum_r U[i,j,r] V[j',k,r] W[k,i,r]
    # We want C[i, k, j, j] = 1 and C[i, k, j, j'] = 0 for j != j'
    
    # Efficient computation:
    # For each i, k:
    #   W_vec = W[k, i, :] (shape R)
    #   U_i = U[i, :, :] (shape m, R)
    #   V_k = V[:, k, :] (shape m, R) -- Note V is (m, p, R), so V[:, k, :] is (m, R)
    #   Matrix M = U_i * W_vec (shape m, R) -> broadcast? No.
    #   M[j, r] = U[i, j, r] * W[k, i, r]
    #   Then C[i, k, j, j'] = sum_r M[j, r] * V[j', k, r]
    #   This is M @ V_k.T (shape m, m)
    
    for i in range(n):
        for k in range(p):
            w_vec = W3[k, i, :]
            u_i = U3[i, :, :] # (m, R)
            v_k = V3[:, k, :] # (m, R)
            
            # M[j, r] = u_i[j, r] * w_vec[r]
            M = u_i * w_vec # (m, R)
            
            # C_jj' = sum_r M[j, r] * v_k[j', r]
            C_mat = M @ v_k.T # (m, m)
            
            # Check C_mat
            # Diagonal should be 1
            if not np.allclose(np.diag(C_mat), 1.0, atol=1e-9):
                return False
            # Off-diagonal should be 0
            # Create mask
            mask = np.ones((m, m), dtype=bool)
            np.fill_diagonal(mask, False)
            if not np.allclose(C_mat[mask], 0.0, atol=1e-9):
                return False
    return True

def _solve_w(U, V, n, m, p):
    """
    Given U and V, solve for W.
    Returns W if a solution exists that satisfies diagonal constraints.
    """
    R = U.shape[1]
    W = np.zeros((p, n, R))
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    
    for i in range(n):
        for k in range(p):
            # System: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for j=0..m-1
            # Let A[j, r] = U[i,j,r] * V[j,k,r]
            A = U3[i, :, :] * V3[:, k, :] # (m, R)
            b = np.ones(m)
            
            # Solve A @ w = b
            # A is m x R. w is R x 1.
            if m > R:
                # Underdetermined? No, m=5, R>=5 usually.
                # If R < m, no unique solution, but we can use lstsq.
                pass
            
            # Use least squares
            w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if it's an exact solution
            if np.allclose(A @ w, b, atol=1e-9):
                W[k, i, :] = w
            else:
                return None
    return W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Schoolbook
    R_school = n * m * p
    U = np.zeros((n * m, R_school))
    V = np.zeros((m * p, R_school))
    W = np.zeros((p * n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = R_school
    
    # Search for lower rank
    # Try ranks from 99 down to 40
    for r in range(99, 39, -1):
        if time.time() > deadline - 5:
            break
            
        # Number of iterations depends on time
        start_iter_time = time.time()
        max_iters = 5000 if r > 80 else 20000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Random sparse U, V
            # Density 0.5? Or 0.3?
            # Entries in {-1, 0, 1}
            density = 0.5
            U_rand = np.zeros((n * m, r))
            V_rand = np.zeros((m * p, r))
            
            mask_u = np.random.rand(n * m, r) < density
            mask_v = np.random.rand(m * p, r) < density
            
            U_rand[mask_u] = np.random.choice([-1, 1], size=np.sum(mask_u))
            V_rand[mask_v] = np.random.choice([-1, 1], size=np.sum(mask_v))
            
            # Solve for W
            W_sol = _solve_w(U_rand, V_rand, n, m, p)
            
            if W_sol is not None:
                # Verify strong condition
                if _verify_strong(U_rand, V_rand, W_sol, n, m, p):
                    best = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_sol.tolist()
                    }
                    best_rank = r
                    break
                    
    return best
