import scipy
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Target tensor T is defined by T[a, b, c] = 1 if indices match, else 0.
    # We can represent T implicitly.
    # a = i*m + j
    # b = j'*p + k
    # c = k'*n + i'
    # Match if j == j', i == i', k == k'.
    
    # Precompute index mappings for faster lookup
    # a -> (i, j)
    a_idx = [(a // m, a % m) for a in range(nm)]
    # b -> (j, k)
    b_idx = [(b // p, b % p) for b in range(mp)]
    # c -> (k, i)
    c_idx = [(c // n, c % n) for c in range(pn)]
    
    # We want to minimize || sum_r U[:,r] V[:,r]^T W[:,r]^T - T ||^2
    # This is equivalent to minimizing the sum of squares of errors for each (a,b,c).
    
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Try to find lower ranks
    # We'll try ranks from 89 down to 80 (or until time runs out)
    # Using a simple gradient descent or L-BFGS-B
    
    # Since the problem is non-convex, we need good initialization.
    # Initializing with schoolbook + noise is a good start.
    
    def compute_error(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Result R_ab_c = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this as:
        # M = U @ V.T  (nm x mp) -- No, this sums over r.
        # We need sum_r U[a,r] V[b,r] W[c,r].
        # This is not a simple matrix multiplication.
        # It is a tensor contraction.
        # For each r, we have a rank-1 tensor U[:,r] (nm) x V[:,r] (mp) x W[:,r] (pn).
        # Summing them gives a 3D tensor.
        # We only need to compare with T which is sparse.
        # T[a,b,c] is 1 only if indices match.
        # Number of non-zeros in T is n*m*p = 90.
        # Specifically, for each (i,j,k), a = i*m+j, b = j*p+k, c = k*n+i.
        # T[a,b,c] = 1.
        # All other T[a,b,c] = 0.
        
        # So we only need to check the 90 non-zero entries and ensure they are 1.
        # And we need to ensure the other entries are 0?
        # Yes, "exactly decompose".
        # So we need sum_r U[a,r] V[b,r] W[c,r] = 1 for the 90 valid triples.
        # And sum_r U[a,r] V[b,r] W[c,r] = 0 for all other triples.
        
        # The number of "other" triples is nm*mp*pn - 90 = 4860 - 90 = 4770.
        # Checking all 4860 is feasible.
        
        error = 0.0
        # Check non-zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.dot(np.dot(U[:, :R].T, V[:, :R]), W[:, :R]) # This is wrong shape
                    # val = sum_r U[a,r] V[b,r] W[c,r]
                    # We can compute this vector for all (a,b,c) at once?
                    pass
        # This loop is too slow if done in Python.
        # We need a vectorized approach.
        
        # Let's compute the full tensor sum.
        # S = sum_r U[:,r] (nm) * V[:,r] (mp) * W[:,r] (pn)
        # S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute S using einsum.
        # S = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a tensor of size nm x mp x pn = 4860.
        # This is small enough to compute directly.
        
        S = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Construct target T (sparse)
        # T is zero everywhere, except 90 spots.
        # We can create a zero array and set the 90 spots.
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        
        error = np.sum((S - T)**2)
        return error

    # Gradient descent
    # We need the gradient of error w.r.t U, V, W.
    # Error = sum (S - T)^2
    # dError/dS = 2(S - T)
    # dS/du_ar = V_br W_cr
    # dS/dv_br = U_ar W_cr
    # dS/dw_cr = U_ar V_br
    
    # We can use scipy.optimize.minimize with L-BFGS-B.
    
    def objective(params, R):
        # params: flattened U, V, W
        U = params[0:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:nm*R+mp*R+pn*R].reshape(pn, R)
        S = np.einsum('ar,br,cr->abc', U, V, W)
        # T is precomputed
        return np.sum((S - T)**2)

    # Precompute T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    # Try ranks
    for R in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Initialize
            # Use schoolbook as base, but it has rank 90.
            # We need to project it to rank R?
            # Or just random init.
            # Random init might be slow.
            # Let's try initializing with the first R columns of schoolbook.
            U_init = U_school[:, :R].copy()
            V_init = V_school[:, :R].copy()
            W_init = W_school[:, :R].copy()
            
            # Add noise
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = scipy.optimize.minimize(objective, x0, args=(R,), method='L-BFGS-B', 
                                              options={'maxiter': 100, 'ftol': 1e-10})
                if res.fun < 1e-8:
                    # Found a solution!
                    U_sol = res.x[0:nm*R].reshape(nm, R)
                    V_sol = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_sol = res.x[nm*R+mp*R:nm*R+mp*R+pn*R].reshape(pn, R)
                    # Verify
                    S_sol = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                    if np.allclose(S_sol, T, atol=1e-8):
                        best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                        best_rank = R
                        # We can break out of loops if we find a good one, 
                        # but maybe we can go lower?
                        # The loop continues to lower R.
                        break
            except Exception:
                pass

    return best_result
