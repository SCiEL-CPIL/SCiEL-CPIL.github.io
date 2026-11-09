"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = 145.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T
    # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
    # T is 1 if indices match, 0 otherwise.
    # Dimensions: (n*m, m*p, n*p) = (6, 24, 16)
    
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook decomposition
    # U: (n*m, rank), V: (m*p, rank), W: (n*p, rank)
    # For each (i, j, k), we have a term.
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((n * p, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    def verify(U, V, W, tol=1e-9):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        # Efficient check:
        # For each c, check if T[:, :, c] == sum_r U[:, r] * V[:, r] * W[c, r]
        # This is a linear combination of rank-1 matrices U[:,r] V[:,r]^T.
        # Let M_c = T[:, :, c].
        # We need M_c = sum_r W[c, r] * (U[:, r] outer V[:, r]).
        
        # Precompute U[:, r] outer V[:, r] for all r?
        # That's a 3D tensor of size (n*m, m*p, rank).
        # Then for each c, we do a matrix product.
        
        # Alternatively, iterate over r and accumulate.
        # Error = sum_c || T[:, :, c] - sum_r W[c, r] * (U[:, r] outer V[:, r]) ||^2
        
        # Let's just do the direct check for small dimensions.
        # n*m = 6, m*p = 24, n*p = 16.
        # 6 * 24 * 16 = 2304 entries.
        # rank is ~48.
        # 2304 * 48 ~ 110k operations. Very fast.
        
        err = 0.0
        for c in range(n * p):
            # Compute the reconstructed slice
            # Rec_c = sum_r W[c, r] * U[:, r] * V[:, r]
            # This is an outer product sum.
            # We can do: Rec_c = (U * W[c, :]) @ V.T
            # U is (6, r), W[c, :] is (r,), so U * W[c, :] is (6, r).
            # V is (24, r), V.T is (r, 24).
            # Result is (6, 24).
            Rec = (U * W[c, :]) @ V.T
            err += np.sum((T[:, :, c] - Rec) ** 2)
            
        return err < tol

    # First, verify schoolbook
    if not verify(U_sb, V_sb, W_sb):
        # This should not happen
        pass
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition
    # We will try ranks from 47 down to 40 (arbitrary limit for search)
    # Using a simple alternating minimization approach.
    
    # We can try to "merge" columns in the schoolbook decomposition.
    # Or just random search.
    
    # Let's try a more structured approach:
    # The tensor is separable in some ways.
    # But for now, let's try a random search for rank 47.
    
    # If we find a valid decomposition of rank r, we update best.
    
    # Given the difficulty of finding an exact decomposition, 
    # and the high probability that 48 is optimal for these dimensions 
    # (or close to it), we might not find a better one quickly.
    
    # However, let's try to reduce the rank by 1 or 2 if possible.
    
    # Strategy:
    # 1. Start with schoolbook U, V, W.
    # 2. Randomly select two columns to merge? No, that's not how it works.
    # 3. Use a gradient-based method to minimize the error of a rank-r approximation.
    
    # Let's try rank 47.
    # Initialize U, V, W with the first 47 columns of schoolbook.
    # Add small random noise to avoid local minima.
    # Run a few iterations of ALS.
    
    def find_lower_rank(target_rank, max_iters=100):
        if target_rank >= schoolbook_rank:
            return None
        
        # Initialize
        U = U_sb[:, :target_rank].copy()
        V = V_sb[:, :target_rank].copy()
        W = W_sb[:, :target_rank].copy()
        
        # Add noise
        noise_scale = 0.1
        U += noise_scale * np.random.randn(*U.shape)
        V += noise_scale * np.random.randn(*V.shape)
        W += noise_scale * np.random.randn(*W.shape)
        
        for it in range(max_iters):
            # Update U
            # For each a, U[a, :] = argmin_Ua sum_c (T[a, :, c] - sum_r U[a, r] V[:, r] W[c, r])^2
            # This is a linear least squares problem for each a.
            # Let M_a = T[a, :, :] (shape m*p, n*p)
            # We want M_a = sum_r U[a, r] (V[:, r] outer W[:, r])
            # Let Z_r = V[:, r] outer W[:, r] (shape m*p, n*p)
            # Then M_a = sum_r U[a, r] Z_r
            # This is equivalent to vectorizing M_a and Z_r.
            # vec(M_a) = sum_r U[a, r] vec(Z_r)
            # So we solve a linear system: A U[a, :]^T = vec(M_a)
            # where A is a matrix with columns vec(Z_r).
            
            # Construct A (shape m*p*n*p, rank)
            # This is expensive to construct every time.
            # But dimensions are small: 24 * 16 = 384. rank ~ 47.
            # 384 x 47 matrix.
            
            A = np.zeros((m * p * n * p, target_rank))
            for r in range(target_rank):
                Z_r = np.outer(V[:, r], W[:, r])
                A[:, r] = Z_r.flatten()
                
            # Solve for each a
            for a in range(n * m):
                b = T[a, :, :].flatten()
                # Least squares
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
            # Update V
            # For each b, V[b, :] = argmin_Vb sum_c (T[:, b, c] - sum_r U[:, r] V[b, r] W[c, r])^2
            # Let M_b = T[:, b, :] (shape n*m, n*p)
            # M_b = sum_r V[b, r] (U[:, r] outer W[:, r])
            # vec(M_b) = sum_r V[b, r] vec(U[:, r] outer W[:, r])
            
            A = np.zeros((n * m * n * p, target_rank))
            for r in range(target_rank):
                Z_r = np.outer(U[:, r], W[:, r])
                A[:, r] = Z_r.flatten()
                
            for b in range(m * p):
                b_vec = T[:, b, :].flatten()
                V[b, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                
            # Update W
            # For each c, W[c, :] = argmin_Wc sum_a,b (T[a, b, c] - sum_r U[a, r] V[b, r] W[c, r])^2
            # Let M_c = T[:, :, c] (shape n*m, m*p)
            # M_c = sum_r W[c, r] (U[:, r] outer V[:, r])
            # vec(M_c) = sum_r W[c, r] vec(U[:, r] outer V[:, r])
            
            A = np.zeros((n * m * m * p, target_rank))
            for r in range(target_rank):
                Z_r = np.outer(U[:, r], V[:, r])
                A[:, r] = Z_r.flatten()
                
            for c in range(n * p):
                c_vec = T[:, :, c].flatten()
                W[c, :], residuals, rank, sv = np.linalg.lstsq(A, c_vec, rcond=None)
                
            # Check error
            if verify(U, V, W, tol=1e-6):
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                if verify(U_snapped, V_snapped, W_snapped, tol=1e-9):
                    return U_snapped, V_snapped, W_snapped
                else:
                    return U, V, W
                    
            if time.time() > deadline - 10:
                return None
                
        return None

    # Search for lower ranks
    # Try 47, 46, 45
    for r in range(schoolbook_rank - 1, max(30, schoolbook_rank - 10), -1):
        if time.time() > deadline - 5:
            break
            
        res = find_lower_rank(r, max_iters=50)
        if res is not None:
            U, V, W = res
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break # Found a lower rank, stop searching lower? 
                # Actually, we might find an even lower one, but let's be safe.
                # Or continue to try lower?
                # Let's continue to try lower ranks to see if we can go even lower.
                # But we need to be careful with time.
                
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
