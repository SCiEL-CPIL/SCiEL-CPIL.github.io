import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Define the target tensor T_{i,j,k}
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In matrix multiplication C = A @ B, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is 1 for all valid (i,j,k).
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # 2. Define the schoolbook solution as a fallback (Rank n*m*p = 100)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return {'U': U, 'V': V, 'W': W}

    best_solution = None

    # 3. Randomized Search for lower rank
    # We attempt to find a decomposition of rank r < 100.
    # We use a "tensor completion" like approach or simple gradient descent.
    # Given U, V, W, the residual is R = T - sum_r U_r \otimes V_r \otimes W_r.
    # We want to minimize ||R||_F^2.
    
    def evaluate_decomposition(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
        # Note: indices in U are flattened (i*m+j), V (j*p+k), W (k*n+i)
        # Let's verify a few points or all points if fast enough.
        # For n=4, m=5, p=5, total points 100. Rank up to 100.
        # 100 * 100 ops = 10,000 ops. Very fast.
        r = U.shape[1]
        # Reshape for broadcasting?
        # U_r is vector of length n*m. 
        # It is easier to loop over r and accumulate.
        # T_reconstructed = np.zeros((n, m, p))
        # for r_idx in range(r):
        #     u = U[:, r_idx].reshape(n, m)
        #     v = V[:, r_idx].reshape(m, p)
        #     w = W[:, r_idx].reshape(p, n)
        #     # We need T[i,j,k] = sum u[i,j] v[j,k] w[k,i]
        #     # This is a contraction.
        #     # Let's just check the condition directly.
        
        # Actually, let's just compute the error.
        # Error[i,j,k] = 1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Efficient check:
        # We can check if the max absolute error is close to 0.
        # Since we use integer/half-integer entries ideally, but search might find floats.
        # The problem requires integer/half-integer.
        # So after finding a float solution, we must round.
        
        # Let's implement a function that checks validity of a specific U,V,W (floats)
        # and returns the max error.
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    err = abs(1.0 - s)
                    if err > max_err:
                        max_err = err
        return max_err

    def round_to_half_int(M):
        # Round to nearest 0.5
        return np.round(M * 2) / 2.0

    def try_rank(r):
        # Initialize random U, V, W
        # To help convergence, we can start with a rank-r projection of the schoolbook solution
        # Or just random.
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Gradient Descent
        # Objective: F(U,V,W) = 0.5 * || T - sum U_r V_r W_r ||_F^2
        # This is non-convex.
        
        # We will use a simple iterative refinement.
        # Fix V, W, solve for U (linear least squares).
        # Fix U, W, solve for V.
        # Fix U, V, solve for W.
        
        # This is the "Alternating Least Squares" (ALS) method for tensor decomposition.
        # It converges to a local minimum.
        
        # Pre-compute some indices for speed?
        # The update for U:
        # U[:, r] should minimize sum_{i,j,k} (T[i,j,k] - sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r'])^2
        # This is a linear system in U[:, r].
        # Let's define the "unfolding" or just use the normal equations.
        
        # For fixed r, let u = U[:, r].
        # The term involving u in the residual is - u[i*m+j] * V[j*p+k, r] * W[k*n+i, r].
        # This couples all i,j,k.
        
        # A simpler approach: Random restarts with low-rank initialization.
        # Given the small size, we can try many random initializations.
        
        # Let's try a few random restarts.
        for restart in range(50):
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                # For each r, we want to find u_r that best fits the residual given current V, W.
                # Residual R[i,j,k] = T[i,j,k] - sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r']
                # We want to minimize sum_{i,j,k} (R[i,j,k] + u_r[i*m+j] V[j,k,r] W[k,i,r])^2
                # This is equivalent to:
                # sum_{i,j,k} (Target[i,j,k] - u_r[i*m+j] * M_r[j,k,i])^2
                # where Target = T - sum_{r' != r} U[i,j,r'] V[j,k,r'] W[k,i,r']
                # and M_r[j,k,i] = V[j,k,r] W[k,i,r]
                
                # This is a least squares problem:
                # min_u || Target_vec - A_r u ||^2
                # where A_r is a matrix constructed from M_r.
                
                # Constructing A_r explicitly might be memory heavy or slow if done naively.
                # Dimensions:
                # u has size n*m = 20.
                # Target has size n*m*p = 100.
                # A_r will be 100 x 20.
                
                # A_r[row, col] where row corresponds to (i,j,k) and col to (i',j')
                # A_r[(i,j,k), (i',j')] = V[j,k,r] W[k,i,r] if (i,j) == (i',j') else 0?
                # No, the term is u[i*m+j] * V[j,k,r] * W[k,i,r].
                # So for a fixed (i,j,k), the coefficient of u[i*m+j] is V[j,k,r] W[k,i,r].
                # The coefficient of u[i'*m+j'] is 0 if (i',j') != (i,j).
                # So A_r is block diagonal? No, it's sparse.
                # Actually, for each (i,j), the variable u[i*m+j] appears in all k terms.
                # So we can solve for each (i,j) independently?
                # No, the objective is sum over all i,j,k.
                # Let's group by (i,j).
                # For a fixed (i,j), let u_val = u[i*m+j].
                # The terms involving u_val are for all k:
                # sum_k (T[i,j,k] - u_val V[j,k,r] W[k,i,r] - OtherTerms[i,j,k])^2
                # where OtherTerms = sum_{r' != r} U[i,j,r'] V[j,k,r'] W[k,i,r']
                
                # So for each (i,j), we have a 1D least squares problem!
                # min_u_val sum_k (Y_k - u_val X_k)^2
                # where Y_k = T[i,j,k] - OtherTerms[i,j,k]
                # and X_k = V[j,k,r] W[k,i,r]
                
                # Solution: u_val = (sum Y_k X_k) / (sum X_k^2)
                
                # Let's implement this.
                
                # First, compute the "OtherTerms" or just the current full sum.
                # Let S[i,j,k] = sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r']
                # We can compute S efficiently.
                
                # S is shape (n,m,p)
                S = np.zeros((n, m, p))
                for r_idx in range(r):
                    u = U[:, r_idx].reshape(n, m)
                    v = V[:, r_idx].reshape(m, p)
                    w = W[:, r_idx].reshape(p, n)
                    # S[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                    # This is not a standard matmul.
                    # We can loop or use einsum.
                    # np.einsum('ij,jk,ki->ijk', u, v, w)
                    S += np.einsum('ij,jk,ki->ijk', u, v, w)
                    
                # Now for each r, update U[:, r]
                for r_idx in range(r):
                    v = V[:, r_idx].reshape(m, p)
                    w = W[:, r_idx].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            # X_k = v[j,k] * w[k,i]
                            X = v[j, :] * w[:, i] # shape (p,)
                            # Y_k = T[i,j,k] - (S[i,j,k] - U[i,j,r_idx]*X_k)
                            # S[i,j,k] includes the term for r_idx.
                            # Let S_minus_r = S - U[:, r_idx].reshape(n,m)[:, :, None] * (v * w.T) # careful with shapes
                            
                            # Actually, easier:
                            # Current contribution of r_idx is C_r = U[i,j,r_idx] * X
                            # Other contributions = S[i,j,:] - C_r
                            # Y = T[i,j,:] - (S[i,j,:] - C_r) = T[i,j,:] - S[i,j,:] + C_r
                            # Wait, we want to fit T with the new u_val.
                            # Residual for other terms is R_other = T - (S - C_r)
                            # We want to minimize || R_other - u_val_new * X ||^2
                            # So Y_k = T[i,j,k] - (S[i,j,k] - U[i,j,r_idx]*X_k)
                            
                            Y = T[i, j, :] - (S[i, j, :] - U[i, j, r_idx] * X)
                            
                            denom = np.dot(X, X)
                            if denom < 1e-12:
                                # If X is zero, u_val doesn't matter, keep old or 0
                                U[i, j, r_idx] = 0.0
                            else:
                                U[i, j, r_idx] = np.dot(Y, X) / denom
                
                # Similarly for V and W
                # Update V[:, r_idx]
                for r_idx in range(r):
                    u = U[:, r_idx].reshape(n, m)
                    w = W[:, r_idx].reshape(p, n)
                    
                    for j in range(m):
                        for k in range(p):
                            # Variable is v_val = V[j,k,r_idx]
                            # Term is u[i,j] * v_val * w[k,i]
                            # X_i = u[i,j] * w[k,i] for i in 0..n-1
                            X = u[:, j] * w[k, :] # shape (n,)
                            
                            # Y_i = T[i,j,k] - (S[i,j,k] - V[j,k,r_idx]*X_i)
                            # Note: S[i,j,k] depends on i.
                            # We are minimizing over i.
                            Y = T[:, j, k] - (S[:, j, k] - V[j, k, r_idx] * X)
                            
                            denom = np.dot(X, X)
                            if denom < 1e-12:
                                V[j, k, r_idx] = 0.0
                            else:
                                V[j, k, r_idx] = np.dot(Y, X) / denom
                
                # Update W[:, r_idx]
                for r_idx in range(r):
                    u = U[:, r_idx].reshape(n, m)
                    v = V[:, r_idx].reshape(m, p)
                    
                    for k in range(p):
                        for i in range(n):
                            # Variable is w_val = W[k,i,r_idx]
                            # Term is u[i,j] * v[j,k] * w_val
                            # X_j = u[i,j] * v[j,k] for j in 0..m-1
                            X = u[i, :] * v[:, k] # shape (m,)
                            
                            # Y_j = T[i,j,k] - (S[i,j,k] - W[k,i,r_idx]*X_j)
                            Y = T[i, :, k] - (S[i, :, k] - W[k, i, r_idx] * X)
                            
                            denom = np.dot(X, X)
                            if denom < 1e-12:
                                W[k, i, r_idx] = 0.0
                            else:
                                W[k, i, r_idx] = np.dot(Y, X) / denom

            # After ALS, check error
            err = evaluate_decomposition(U, V, W)
            if err < 1e-5:
                # Round to half-integers
                U_int = round_to_half_int(U)
                V_int = round_to_half_int(V)
                W_int = round_to_half_int(W)
                
                # Verify integer solution
                err_int = evaluate_decomposition(U_int, V_int, W_int)
                if err_int < 1e-9:
                    return {'U': U_int, 'V': V_int, 'W': W_int}
                else:
                    # Maybe rounding broke it, try nearest integers?
                    U_int2 = np.round(U)
                    V_int2 = np.round(V)
                    W_int2 = np.round(W)
                    err_int2 = evaluate_decomposition(U_int2, V_int2, W_int2)
                    if err_int2 < 1e-9:
                        return {'U': U_int2, 'V': V_int2, 'W': W_int2}
            else:
                # If error is small but not zero, maybe we can fix it?
                # For now, just record if it's better than current best
                pass

        return None

    # Try ranks from 60 up to 99
    # We know 100 works.
    # Theoretical lower bound for 4x5x5 is not known exactly, but likely < 100.
    # Strassen 4x4x4 is 49.
    # Let's try a few ranks.
    
    # To save time, we can try a specific rank like 80, 70, 60.
    # If we find one, we return it.
    
    # However, ALS might get stuck.
    # Let's try the schoolbook first to ensure we have a valid result.
    # Then try to improve.
    
    schoolbook = get_schoolbook()
    
    # Try a few random ranks
    ranks_to_try = [90, 80, 70, 60]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        sol = try_rank(r)
        if sol is not None:
            return sol
            
    return schoolbook

