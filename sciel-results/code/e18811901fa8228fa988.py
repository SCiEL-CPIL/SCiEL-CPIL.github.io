import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    start_time = time.time()
    deadline = start_time + 110 # Leave 40s margin
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Greedy Tensor Rank Decomposition Search
    # We try to find a decomposition of rank R starting from a high rank 
    # and trying to reduce it, or starting from low and adding.
    # Given the small dimensions, we can try to find the minimal rank by 
    # checking if a rank R decomposition exists using a greedy additive approach.
    
    # The theoretical lower bound for matrix multiplication rank is roughly n*m*p / (some factor).
    # For 3x5x6, schoolbook is 90.
    # Known results: R(3,3,3)=23, R(2,2,2)=7.
    # Let's try ranks from 50 upwards. If we find one, we record it and try to go lower.
    
    # However, finding the MINIMAL rank is hard. 
    # The task says "Find a rank-minimal exact tensor decomposition".
    # But the evaluator scores "raw value" (rank). Lower is better.
    # So we want the smallest rank possible.
    
    # Strategy: 
    # 1. Try to find a decomposition for rank R.
    # 2. If found, check if we can go lower.
    # 3. Return the best (lowest) rank found.
    
    # We will use a greedy algorithm that builds U, V, W column by column.
    # At each step, we pick the column that minimizes the remaining error.
    
    # To make it efficient, we pre-compute the "structure" indices.
    # Indices for U: (i, j) -> i*m + j
    # Indices for V: (j, k) -> j*p + k
    # Indices for W: (k, i) -> k*n + i
    
    # The tensor T is defined by T[i, j, k] = 1 for all i, j, k.
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    # Greedy approach:
    # Start with residual R = T.
    # For r in 1..MaxRank:
    #   Find u, v, w that best approximate R.
    #   R -= u * v * w
    #   If R is close to 0, we have a decomposition of rank r.
    
    # How to find best u, v, w for a given residual R?
    # This is a 3-way tensor decomposition problem.
    # We can use ALS (Alternating Least Squares) on the residual.
    
    def _greedy_find_rank(max_rank, n, m, p, time_limit=10.0):
        """
        Attempts to find a decomposition of rank <= max_rank.
        Returns (U, V, W) if found, else None.
        """
        start = time.time()
        
        # Initialize residual
        # We represent the residual implicitly or explicitly?
        # Explicit tensor is 3*5*6 = 90 entries. Small enough to keep explicit.
        # T is all 1s.
        residual = np.ones((n, m, p))
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        for r in range(1, max_rank + 1):
            if time.time() - start > time_limit:
                break
            
            # Find best rank-1 approximation of residual
            # We use a few random restarts for ALS on the residual
            best_u, best_v, best_w = None, None, None
            best_err = np.inf
            
            for trial in range(5):
                # Random init
                u = np.random.randn(n, m)
                v = np.random.randn(m, p)
                w = np.random.randn(p, n)
                
                # Normalize to avoid scale issues
                # The product u*v*w should sum to 1.
                # We can fix the norm of one tensor, say w.
                norm_w = np.linalg.norm(w)
                if norm_w > 0:
                    w = w / norm_w
                else:
                    continue
                
                # ALS iterations
                for _ in range(50):
                    # Update u
                    # For each i, j: sum_k u[i,j] * v[j,k] * w[k,i] = residual[i,j,k]
                    # This is: u[i,j] * sum_k v[j,k] w[k,i] = residual[i,j,k]
                    # This is not a simple linear system for u because v and w are coupled.
                    # Wait, the equation is:
                    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
                    # If we are adding one rank at a time, we are solving for the single rank term.
                    # Actually, the standard CP decomposition minimizes ||T - sum U_r V_r W_r||^2.
                    # We want exact match, so we want ||Residual - u v w||^2 = 0.
                    
                    # Let's use a simpler greedy step:
                    # Pick a random (i, j, k) where residual is non-zero?
                    # No, better to use ALS on the whole tensor.
                    
                    # Fix v, w -> solve for u
                    # u[i,j] * (v[j,:] * w[:,i]).sum() = residual[i,j,:] .sum()? No.
                    # The term is u[i,j] * v[j,k] * w[k,i].
                    # For a fixed i, j, the equation over k is:
                    # u[i,j] * sum_k v[j,k] w[k,i] = sum_k residual[i,j,k] (if we project?)
                    # No, we want element-wise match.
                    # u[i,j] v[j,k] w[k,i] approx residual[i,j,k] for all k.
                    # This is a rank-1 approximation of the matrix slice residual[i, :, :]? No.
                    
                    # Let's reformulate.
                    # We want to minimize sum_{i,j,k} (residual[i,j,k] - u[i,j]v[j,k]w[k,i])^2.
                    # This is the standard CP decomposition objective.
                    
                    # Update U:
                    # For fixed i, j:
                    # Let M_{ij} = sum_k residual[i,j,k] * v[j,k] * w[k,i]
                    # Let D_{ij} = sum_k (v[j,k] * w[k,i])^2
                    # u[i,j] = M_{ij} / D_{ij}
                    
                    U_new = np.zeros((n, m))
                    for i in range(n):
                        for j in range(m):
                            # v[j, :] is (p,), w[:, i] is (p,)
                            vec = v[j, :] * w[:, i]
                            D = np.dot(vec, vec)
                            if D > 1e-12:
                                M = np.dot(residual[i, j, :], vec)
                                U_new[i, j] = M / D
                            else:
                                U_new[i, j] = 0
                    u = U_new
                    
                    # Update V:
                    V_new = np.zeros((m, p))
                    for j in range(m):
                        for k in range(p):
                            # u[:, j] is (n,), w[k, :] is (n,)
                            vec = u[:, j] * w[k, :]
                            D = np.dot(vec, vec)
                            if D > 1e-12:
                                M = np.dot(residual[:, j, k], vec)
                                V_new[j, k] = M / D
                            else:
                                V_new[j, k] = 0
                    v = V_new
                    
                    # Update W:
                    W_new = np.zeros((p, n))
                    for k in range(p):
                        for i in range(n):
                            # u[i, :] is (m,), v[:, k] is (m,)
                            vec = u[i, :] * v[:, k]
                            D = np.dot(vec, vec)
                            if D > 1e-12:
                                M = np.dot(residual[i, :, k], vec)
                                W_new[k, i] = M / D
                            else:
                                W_new[k, i] = 0
                    w = W_new
                    
                    # Normalize w to unit norm, absorb scale into u, v?
                    # Actually, for exact decomposition, scaling matters.
                    # Let's keep the scales as found.
                    
                # Check error
                approx = np.einsum('ij,jk,ki->ijk', u, v, w)
                err = np.linalg.norm(residual - approx)
                
                if err < best_err:
                    best_err = err
                    best_u, best_v, best_w = u, v, w
                    
            # If best error is small, accept this rank
            if best_err < 1e-6:
                # Update residual
                approx = np.einsum('ij,jk,ki->ijk', best_u, best_v, best_w)
                residual -= approx
                
                # Store columns
                U_cols.append(best_u.flatten())
                V_cols.append(best_v.flatten())
                W_cols.append(best_w.flatten())
                
                # Check if residual is zero
                if np.allclose(residual, 0, atol=1e-6):
                    # We found a decomposition of rank r
                    U_mat = np.column_stack(U_cols)
                    V_mat = np.column_stack(V_cols)
                    W_mat = np.column_stack(W_cols)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                        return U_rounded, V_rounded, W_rounded
                    else:
                        # Rounding failed. Maybe we can adjust?
                        # For now, if unrounded is exact, we might need a different strategy.
                        # But the task requires integer/half-integer.
                        # If we can't round, this rank might not be valid for the output format.
                        # We should probably continue searching or accept a higher rank that rounds better.
                        # However, exact real solutions often don't round nicely.
                        # Let's return the unrounded one? No, interface requires specific types.
                        # The prompt says "integer/half-integer entries".
                        # So we MUST round.
                        # If rounding fails, this candidate is invalid.
                        pass
            
            # If we didn't accept, or error was high, we stop?
            # Greedy usually stops if improvement is small.
            # But for exact decomposition, we need to reach 0.
            # If we can't reach 0 with r ranks, we try r+1.
            
        return None

    # Search for minimal rank
    # We know schoolbook is 90.
    # Let's try to find something better.
    # Start from a reasonable guess, e.g., 60, and go down?
    # Or start from 1 and go up?
    # Starting from 1 is slow if the rank is high.
    # Let's try a few specific ranks that are known to be close or plausible.
    # For 3x5x6, rank is likely around 60-80?
    
    # Let's try ranks: 90, 85, 80, 75, 70, 65, 60, 55, 50
    # If we find 80, we check 79, 78...
    
    # To save time, we will try to find *any* decomposition lower than 90 first.
    # Then refine.
    
    candidate_ranks = [89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    found_rank = None
    found_U, found_V, found_W = None, None, None
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Use a tight time limit per rank
        time_left = deadline - time.time()
        if time_left < 2:
            break
            
        # Allocate time: more time for lower ranks (harder)
        t_limit = min(5.0, time_left * 0.5)
        
        result = _greedy_find_rank(rank, n, m, p, time_limit=t_limit)
        
        if result is not None:
            U, V, W = result
            if found_rank is None or rank < found_rank:
                found_rank = rank
                found_U, found_V, found_W = U, V, W
                
            # If we found a rank, we can try to go lower.
            # But if we found 80, we should check 79, 78...
            # The loop continues, so it will check lower ranks in the list.
            # However, if we found 80, we don't need to check 85, 86, etc.
            # But the list is sorted descending.
            # So if we find 80, we should skip 81-89?
            # No, the list is 89, 88... 50.
            # If we find 80, we have checked 89..81 and failed.
            # So we continue to 79.
            
    if found_rank is not None:
        return {"U": found_U.tolist(), "V": found_V.tolist(), "W": found_W.tolist()}
    
    # If nothing found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum"""
    rank = U.shape[1]
    if rank == 0:
        return False
        
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    U_sub = U[u_idx, :]
    V_sub = V[v_idx, :]
    W_sub = W[w_idx, :]
    
    result = np.einsum('ic,ic,ic->i', U_sub, V_sub, W_sub)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _find_rank(rank, n, m, p, max_time=5.0):
    """Placeholder to keep interface if needed, but we use greedy now."""
    return None
