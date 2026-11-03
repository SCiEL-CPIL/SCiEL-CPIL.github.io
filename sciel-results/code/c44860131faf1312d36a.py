import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    tensor = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Flatten tensor for easy comparison
    tensor_flat = tensor.ravel()
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if U,V,W decompose the tensor using vectorized ops."""
        r = U_mat.shape[1]
        recon = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
        if np.max(np.abs(recon - tensor)) > tol:
            return False
        return True

    assert verify_fast(U, V, W), "Schoolbook failed verification"

    # Optimization strategy: ALS (Alternating Least Squares)
    # We try to find a decomposition of rank R < best_rank.
    # We iterate R from some lower bound up to best_rank - 1.
    # For each R, we try several random initializations.
    
    # Lower bound for rank?
    # Rank is at least max(nm, mp, pn) = max(9, 21, 21) = 21.
    # Also rank is at least the "quantum rank" etc.
    # Let's try R from 21 to best_rank - 1.
    
    # To save time, we will try a few specific R values first, then sweep.
    # Or just sweep from a reasonable guess.
    # Let's start from best_rank - 1 and go down.
    
    # Pre-compute flattened slices for faster ALS steps?
    # T is (9, 21, 21).
    # T_1 is (9, 441) where each row is T[i] flattened.
    # T_2 is (21, 189) where each row is T[:, j, :] flattened? No.
    # Let's define the modes properly.
    # Mode 1: nm (9)
    # Mode 2: mp (21)
    # Mode 3: pn (21)
    
    # Reshape tensor to matrices for ALS
    # M1: (nm, mp*pn) -> 9 x 441
    M1 = tensor.reshape(nm, mp * pn)
    # M2: (mp, nm*pn) -> 21 x 189
    M2 = tensor.transpose(1, 0, 2).reshape(mp, nm * pn)
    # M3: (pn, nm*mp) -> 21 x 189
    M3 = tensor.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    def run_als(rank, max_iter=50, tol=1e-8, seed=0):
        """
        Try to find a rank 'rank' decomposition using ALS.
        Returns (U, V, W, error) or None if failed.
        """
        if rank <= 0:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W randomly
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_error = np.inf
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            # Update U
            # T = sum_r U[:,r] V[:,r]^T W[:,r]^T
            # For fixed V, W, T is linear in U.
            # M1 = U @ (V * W).T ?
            # Let's derive:
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # M1[i, j*p+k] = sum_r U[i, r] (V[j, r] * W[k, r])
            # Let B1[r, j*p+k] = V[j, r] * W[k, r]
            # Then M1 = U @ B1.T
            # So U = M1 @ B1.T @ (B1 @ B1.T)^-1 (Least Squares)
            
            # Construct B1: shape (rank, mp*pn)
            # B1[r, idx] = V[j, r] * W[k, r] where idx = j*p+k
            # We can compute this efficiently:
            # V is (mp, rank), W is (pn, rank)
            # We need outer product of V and W per column r, flattened.
            # This is equivalent to:
            # B1 = np.einsum('jr,kr->jrk', V, W).reshape(rank, mp*pn)
            B1 = np.einsum('jr,kr->jrk', V, W).reshape(rank, mp * pn)
            # Solve U B1.T = M1  => U = M1 @ B1.T @ pinv(B1 @ B1.T)
            # Or simpler: np.linalg.lstsq(B1.T, M1.T).T
            try:
                U, _, _, _ = np.linalg.lstsq(B1.T, M1.T, rcond=None)
            except:
                return None
            
            # Update V
            # M2[j, i*pn+k] = sum_r V[j, r] (U[i, r] * W[k, r])
            # B2[r, i*pn+k] = U[i, r] * W[k, r]
            B2 = np.einsum('ir,kr->irk', U, W).reshape(rank, nm * pn)
            try:
                V, _, _, _ = np.linalg.lstsq(B2.T, M2.T, rcond=None)
            except:
                return None
                
            # Update W
            # M3[k, i*mp+j] = sum_r W[k, r] (U[i, r] * V[j, r])
            # B3[r, i*mp+j] = U[i, r] * V[j, r]
            B3 = np.einsum('ir,jr->ijr', U, V).reshape(rank, nm * mp)
            try:
                W, _, _, _ = np.linalg.lstsq(B3.T, M3.T, rcond=None)
            except:
                return None
                
            # Calculate error
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            error = np.linalg.norm(recon - tensor)
            
            if np.abs(prev_error - error) < tol:
                break
            prev_error = error
            
        # Final verification
        if error < 1e-6: # Loose tolerance for ALS, strict for final
            if verify_fast(U, V, W, tol=1e-8):
                return U, V, W, error
        return None

    # Search for lower rank
    # We know rank 63 works.
    # Let's try to find if rank < 63 works.
    # We'll try ranks from 25 to 62.
    # To be safe, we'll start from 62 and go down.
    # If we find a rank R, we continue to R-1.
    
    # Optimization: Use the schoolbook solution as a "warm start" for rank 63?
    # No, random restarts are better for non-convex problems.
    
    # We will allocate time dynamically.
    # Let's try a few random restarts for each rank.
    
    current_best_rank = best_rank
    found_lower = False
    
    # Heuristic: The rank is likely close to the schoolbook rank for these small dimensions unless there's a known algorithm.
    # For 3x3x3, rank is 23 (if true) vs 27.
    # For 3x3x7, maybe it's 63? Or maybe 62?
    # Let's try a few ranks near 63 first, then spread out.
    
    ranks_to_try = list(range(best_rank - 1, 20, -1))
    
    # Limit the number of trials to fit in time
    # Each ALS run takes ~10-50ms.
    # 150s / 0.05s = 3000 runs.
    # We have ~40 ranks to try.
    # So ~75 runs per rank is feasible.
    
    trials_per_rank = 50
    
    for R in ranks_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        success = False
        for trial in range(trials_per_rank):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            res = run_als(R, max_iter=20, seed=trial)
            if res is not None:
                U_sol, V_sol, W_sol, err = res
                if R < current_best_rank:
                    current_best_rank = R
                    best_U = U_sol
                    best_V = V_sol
                    best_W = W_sol
                    success = True
                    # If we found a lower rank, we can break and try to go lower
                    # But maybe we should do a few more trials to ensure it's robust?
                    # The problem asks for *a* decomposition, so one is enough.
                    break
                    
        if not success:
            # If we failed to find rank R, it's likely R is too low.
            # But ALS can get stuck.
            # We continue to next lower R? No, if R fails, R-1 is harder.
            # However, sometimes a specific R fails due to local minima but R-1 works? Unlikely.
            # Usually, if R fails, all lower ranks fail.
            # But let's be careful. We'll stop if we fail for several consecutive ranks.
            # For now, we just continue.
            pass

    # Final check
    if verify_fast(best_U, best_V, best_W):
         return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END
