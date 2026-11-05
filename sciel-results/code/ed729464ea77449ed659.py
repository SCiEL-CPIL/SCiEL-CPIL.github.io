import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Try to find a decomposition of rank 28
    # If not found, return rank 30
    
    def try_rank(R):
        # Initialize U, V, W
        # To help convergence, we can fix some entries or use symmetry breaking
        # Random initialization
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Flatten the tensor equation
        # We want sum_r U[:,r] V[:,r]^T W[:,r]^T = T
        # This is a system of equations.
        # We can use least squares on the flattened tensor.
        
        # Reshape T to 2D: (n*m*m*p, p*n) ? No.
        # T is 3D.
        # We can vectorize the first two modes?
        # Let's define a function to compute the error.
        
        # We will use a simple gradient descent or scipy.optimize.minimize
        # Given the time limit, scipy.optimize might be too slow for 868 variables if not careful.
        # Let's try a few random restarts with a simple local search.
        
        # Actually, for rank 28, the system is very underdetermined.
        # We can set W to be an identity-like matrix or fix it?
        # No, W is part of the solution.
        
        # Let's try to solve using linear algebra if we fix one matrix.
        # Fix W. Then we have T = sum U_r V_r^T W_r^T.
        # This is still non-linear in U and V.
        
        # Alternative: Use the fact that we can permute and scale.
        
        # Let's just use a randomized search with a small number of steps.
        # Or better, use a known decomposition if I can recall it.
        
        # Since I don't have the exact coefficients, I will return the schoolbook rank 30 to ensure validity.
        # However, the prompt asks for a "stronger algorithm" and "structural change".
        # Returning 30 is the baseline.
        # I will attempt a search for rank 28.
        
        # Strategy:
        # 1. Initialize U, V, W.
        # 2. Iterate:
        #    - Fix V, W, solve for U (linear least squares)
        #    - Fix U, W, solve for V (linear least squares)
        #    - Fix U, V, solve for W (linear least squares)
        #    This is a multiplicative update / ALS style.
        
        # Let's implement ALS.
        
        # Reshape T to T_mat of shape (n*m*m*p, p*n)
        # T_mat[ (u*v) + v_idx, w ] = T[u, v, w]
        # This doesn't work directly for ALS because the sum is over r.
        # T[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        
        # If we fix V and W, then for each w:
        # T[:, :, w] = sum_r U[:, r] V[:, r] W[w, r]
        # T[:, :, w] = U (V * W[w, :])
        # Where (V * W[w, :]) is a matrix of size (m*p, m*p)? No.
        # V is (m*p, R). W[w, :] is (R,).
        # The term is U[:, r] V[:, r] W[w, r].
        # This is an outer product of U[:, r] and V[:, r] scaled by W[w, r].
        # So T[:, :, w] = sum_r W[w, r] (U[:, r] V[:, r]^T)
        # This is a linear combination of rank-1 matrices U[:, r] V[:, r]^T.
        # Let M_r = U[:, r] V[:, r]^T.
        # T[:, :, w] = sum_r W[w, r] M_r.
        
        # If we fix U and V, we can solve for W easily?
        # T[:, :, w] = sum_r W[w, r] M_r.
        # This is a linear system for W[:, r] for each w?
        # No, W is (p*n, R).
        # For each w, we have a matrix equation.
        # We can flatten M_r to vectors.
        # Let m_r = vec(M_r).
        # Then vec(T[:, :, w]) = sum_r W[w, r] m_r.
        # This is a linear system for W[w, :] given U, V.
        # We can solve for all W at once by stacking the equations.
        
        # Similarly for U and V.
        
        # Let's implement this ALS.
        
        # Precompute indices
        # T shape (6, 15, 10)
        
        # Initialize
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Normalize to avoid divergence
        for r in range(R):
            norm = np.linalg.norm(U[:, r])
            if norm > 1e-10:
                U[:, r] /= norm
            norm = np.linalg.norm(V[:, r])
            if norm > 1e-10:
                V[:, r] /= norm
            norm = np.linalg.norm(W[:, r])
            if norm > 1e-10:
                W[:, r] /= norm
        
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Solve for W
            # vec(T[:, :, w]) = sum_r W[w, r] vec(U[:, r] V[:, r]^T)
            # Let A be a matrix of shape (n*m*m*p, R) where A[:, r] = vec(U[:, r] V[:, r]^T)
            # Let B be a matrix of shape (p*n, n*m*m*p) where B[w, :] = vec(T[:, :, w])
            # Then B = W^T A  =>  B^T = A^T W
            # We want to solve for W in A^T W = B^T
            # This is a linear least squares problem.
            
            A = np.zeros((n*m * m*p, R))
            for r in range(R):
                M_r = np.outer(U[:, r], V[:, r])
                A[:, r] = M_r.flatten()
            
            # B^T shape (n*m*m*p, p*n)
            B_t = np.zeros((n*m * m*p, p*n))
            for w in range(p*n):
                B_t[:, w] = T[:, :, w].flatten()
            
            # Solve A^T W = B^T
            # A^T is (R, n*m*m*p)
            # W is (R, p*n)
            # B^T is (n*m*m*p, p*n)
            # So we solve A^T X = B^T where X is W^T? No.
            # Equation: sum_r W[w, r] A[:, r] = B_t[:, w]
            # This is A W^T = B_t
            # We want to find W.
            # Let's solve for W^T (shape R, p*n) in A (shape 90, R) W^T (R, 10) = B_t (90, 10)
            # This is a system of 10 independent linear systems?
            # No, A is fixed. We want W^T such that A W^T = B_t.
            # We can solve this using least squares.
            # W^T = (A^T A)^{-1} A^T B_t
            # Or use np.linalg.lstsq(A, B_t)
            
            W_t, residuals, rank, s = np.linalg.lstsq(A, B_t, rcond=None)
            W = W_t.T
            
            # Solve for V
            # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # For fixed u, w:
            # T[u, :, w] = sum_r (U[u, r] W[w, r]) V[:, r]
            # Let C be a matrix of shape (m*p, R) where C[:, r] = V[:, r]
            # We want to find V.
            # T[:, :, w] = sum_r W[w, r] (U[:, r] V[:, r]^T)
            # This is symmetric to W.
            # Let's solve for V.
            # Fix U, W.
            # T[u, v, w] = sum_r V[v, r] (U[u, r] W[w, r])
            # Let D be a matrix of shape (m*p, R) where D[:, r] = V[:, r]
            # Let E be a matrix of shape (n*m*p*n, R) ?
            # Let's flatten the other two indices.
            # T[u, v, w] -> T_flat[(u, w), v]
            # T_flat[(u, w), v] = sum_r V[v, r] (U[u, r] W[w, r])
            # Let F be a matrix of shape (n*m*p*n, R) where F[(u, w), r] = U[u, r] W[w, r]
            # Then T_flat = V^T F
            # We want to solve for V in V^T F = T_flat
            # Or F^T V = T_flat^T
            # F^T is (R, n*m*p*n)
            # V is (R, m*p)
            # T_flat^T is (n*m*p*n, m*p)
            # Solve F^T V = T_flat^T
            
            F = np.zeros((n*m * p*n, R))
            for r in range(R):
                # U[:, r] is (n*m,), W[:, r] is (p*n,)
                # We need the outer product of U[:, r] and W[:, r] flattened
                # But the indices are u and w.
                # F[(u, w), r] = U[u, r] * W[w, r]
                # This is just the outer product of U[:, r] and W[:, r]
                F[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            # T_flat shape (n*m*p*n, m*p)
            T_flat = T.transpose(0, 2, 1).reshape(n*m*p*n, m*p)
            
            # Solve F^T V = T_flat^T
            # F^T is (R, n*m*p*n)
            # V is (R, m*p)
            # T_flat^T is (n*m*p*n, m*p)
            V_t, residuals, rank, s = np.linalg.lstsq(F.T, T_flat.T, rcond=None)
            V = V_t.T
            
            # Solve for U
            # Similarly, solve for U.
            # T[u, v, w] = sum_r U[u, r] (V[v, r] W[w, r])
            # Let G be a matrix of shape (n*m, R) where G[:, r] = U[:, r]
            # Let H be a matrix of shape (m*p*p*n, R) where H[(v, w), r] = V[v, r] W[w, r]
            # Then T_flat2[(u), (v, w)] = sum_r U[u, r] H[(v, w), r]
            # T_flat2 = U^T H
            # H^T U = T_flat2^T
            
            H = np.zeros((m*p * p*n, R))
            for r in range(R):
                H[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            T_flat2 = T.transpose(0, 1, 2).reshape(n*m, m*p*p*n)
            
            H_t, residuals, rank, s = np.linalg.lstsq(H.T, T_flat2.T, rcond=None)
            U = H_t.T
            
            # Check error
            # Reconstruct T
            T_recon = np.zeros_like(T)
            for r in range(R):
                T_recon += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T - T_recon)
            if error < 1e-6:
                return U, V, W, R
        
        return None

    # Try rank 28
    result = try_rank(28)
    if result:
        U, V, W, R = result
        # Convert to lists of lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 30
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
