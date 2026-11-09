"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Build the target tensor T[i,j,k] = 1 for all valid indices
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Target tensor
    T = np.ones((n, m, p))
    
    # Try various ranks from low to high
    # Known lower bounds: for 3x3 x 3x7, rank >= 3*3 = 9 (trivial), 
    # but more carefully, rank >= max(n*m, m*p, p*n) is not quite right
    # For matrix mult (n,m) x (m,p), rank >= n*p (by considering slices)
    # Actually rank >= max(n*m, m*p, p*n) / something... 
    # Let's just search from reasonable lower bounds
    
    # A known result: 3x3 x 3x3 has rank 23 (Strassen-like improvements)
    # For 3x3 x 3x7, we can potentially do better than 63
    
    # Strategy: use random initialization + coordinate descent / alternating minimization
    # with verification at half-integer grid points
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        n, m, p = DIMENSIONS
        # Compute the resulting tensor
        # result[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Use einsum for efficiency
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # Contract: sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # First contract Ur and Vr over j and r: (n, k, ...)
        # Actually: result[i,k] = sum_j sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # This is tricky. Let me do it step by step.
        
        # For each i, k: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    result[i, j, k] = s
        return np.allclose(result, T, atol=tol, rtol=0)
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy operations."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # result[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # Contract Ur and Vr: (i, j, r) x (j, k, r) -> need sum over j and r
        # Let's do: temp[i, j, k, r] = Ur[i,j,r] * Vr[j,k,r]  -- too big
        # Better: for each r, compute outer product structure
        
        # Actually: result[i,k] partial = sum_j sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # For fixed i: sum_j sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # = sum_k Wr[k,i,r] * (sum_j sum_r Ur[i,j,r] * Vr[j,k,r])
        
        # Let me just use a loop over r for clarity and speed
        result = np.zeros((n, m, p))
        for r in range(rank):
            Ur_r = Ur[:, :, r]  # (n, m)
            Vr_r = Vr[:, :, r]  # (m, p)
            Wr_r = Wr[:, :, r]  # (p, n)
            # For each i, j, k: add Ur_r[i,j] * Vr_r[j,k] * Wr_r[k,i]
            # This is: sum_k Wr_r[k,i] * (Ur_r[i,j] * Vr_r[j,k])
            # = Ur_r[i,j] * sum_k Vr_r[j,k] * Wr_r[k,i]
            # Let M[i,j] = Ur_r[i,j], N[j,k] = Vr_r[j,k], P[k,i] = Wr_r[k,i]
            # result_r[i,j,k] = M[i,j] * N[j,k] * P[k,i]
            # Sum over k: for fixed i,j: M[i,j] * sum_k N[j,k]*P[k,i]
            # So for each i: sum_k N[j,k]*P[k,i] for all j -> this is (N @ P)[j,i]
            NP = Vr_r @ Wr_r  # (m, n)
            # Now result_r[i,j,k] needs the k index... 
            # Actually we can't easily collapse k. Let me just do element-wise.
            # For this small problem, explicit loops are fine
            for i in range(n):
                for j in range(m):
                    val = Ur_r[i, j]
                    if val == 0:
                        continue
                    for k in range(p):
                        result[i, j, k] += val * Vr_r[j, k] * Wr_r[k, i]
        
        return np.allclose(result, T, atol=tol, rtol=0)
    
    # Try structured ansätze with specific ranks
    # For 3x3 x 3x7, try ranks from 20 upward
    
    # Method: Random initialization + alternating least squares / coordinate descent
    def try_rank(rank, n_trials=50, max_iter=200, seed_offset=0):
        """Try to find a decomposition of given rank."""
        nonlocal best_U, best_V, best_W, best_rank
        n, m, p = DIMENSIONS
        
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                return False
            
            rng = np.random.RandomState(12345 + seed_offset + trial)
            
            # Initialize with random small values
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
            
            # Scale to make initial residual reasonable
            # Target: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # If we scale U by a, V by b, W by c, the product scales by abc
            # Let's normalize so that typical product is ~1
            
            # Alternating minimization: fix two, solve for third
            # For fixed V, W: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # This is linear in U! For each (i,j), the coefficient of U[i*m+j,r] is V[j*p+k,r]*W[k*n+i,r] summed over k
            # Actually it's: for each i,j: sum_k (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1) * ... 
            # The residual is sum_{i,j,k} (dot(U_row_ij, V_col_jk * W_col_ki) - 1)^2
            # where V_col_jk * W_col_ki means element-wise product of V[j*p+k,:] and W[k*n+i,:]
            
            # For fixed V, W, the problem for U[i*m+j,:] is:
            # minimize sum_k (U[i*m+j,:] · (V[j*p+k,:] * W[k*n+i,:]) - 1)^2
            # Let A_k = V[j*p+k,:] * W[k*n+i,:]  (element-wise product, shape rank)
            # minimize sum_k (U · A_k - 1)^2
            # This is a linear least squares: find U such that U · A_k ≈ 1 for all k
            # Stack: matrix with rows A_k, target vector of 1s
            # U_row = (A^T A)^{-1} A^T * 1
            
            for iteration in range(max_iter):
                if time.time() > deadline - 10:
                    return False
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A_k = V[j*p+k, :] * W[k*n+i, :] for k in 0..p-1
                        A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        # Solve: A @ U_row = ones(p)
                        # Least squares
                        try:
                            U[i * m + j, :], residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # For V[j*p+k, r], it appears in terms where this (j,k) is used
                        # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i, r
                        # For fixed j,k: minimize sum_i (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                        # Let B_i = U[i*m+j, :] * W[k*n+i, :]
                        # minimize sum_i (V_row · B_i - 1)^2
                        B = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        try:
                            V[j * p + k, :], residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # For W[k*n+i, r], appears in terms with this (k,i)
                        # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j, r
                        # minimize sum_j (W_row · C_j - 1)^2 where C_j = U[i*m+j,:] * V[j*p+k,:]
                        C = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        try:
                            W[k * n + i, :], residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Check convergence / validity
                if iteration % 20 == 0 or iteration == max_iter - 1:
                    if verify_fast(U, V, W, tol=1e-8):
                        # Found a valid decomposition!
                        # Try to round to half-integers
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-6):
                            if rank < best_rank:
                                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                                best_rank = rank
                                return True
                        else:
                            # Use the unrounded version if it's close enough
                            if rank < best_rank:
                                best_U, best_V, best_W = U, V, W
                                best_rank = rank
                                return True
            
            # After all iterations, check if we got close
            if verify_fast(U, V, W, tol=1e-7):
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-5):
                    if rank < best_rank:
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = rank
                        return True
                elif rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    return True
        
        return False
    
    # Try ranks from low to high
    # Lower bound: for (n,m,p) matrix mult, rank >= n*p (considering the "output" dimension)
    # Actually a better bound: rank >= max(n*m, m*p, p*n) is not standard
    # For 3x3 x 3x7: n*m=9, m*p=21, p*n=21
    # Known: 3x3 x 3x3 has rank 23, so 3x3 x 3x7 should be >= 23 + something
    # Let's try from 25 upward
    
    for rank in range(25, rank_sb + 1):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            break
        
        n_trials = max(10, 50 - (rank - 25))  # Fewer trials for higher ranks
        success = try_rank(rank, n_trials=n_trials, max_iter=150, seed_offset=rank)
        if success:
            # Found a decomposition at this rank, try to go lower? No, we're going up.
            # But we might find a lower rank later with different initialization
            # Actually we're iterating rank upward, so first success is best
            break
    
    # Convert to lists for output
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
