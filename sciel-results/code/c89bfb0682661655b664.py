"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as guaranteed fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        # Vectorized check
        # For each r, we have outer products
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Let's check using einsum or direct computation
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a bit tricky with einsum. Let's do it step by step.
        # First, for each r, we have a rank-1 tensor U3[:,:,r] ⊗ V3[:,:,r] ⊗ W3[:,:,r]
        # But the indices are coupled: U[i,j], V[j,k], W[k,i]
        
        # Alternative: iterate over r and accumulate
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U3[:, :, r]  # (n, m)
            v = V3[:, :, r]  # (m, p)
            w = W3[:, :, r]  # (p, n)
            # Contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i]
            # This is an outer product with coupled indices
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # We can compute this as: for each i, T[i,:,:] += u[i,:] * (v * w.T)
            # Wait, v[j,k] * w[k,i] is (v * w.T)[j,i]
            # So T[i,j,k] += u[i,j] * (v * w.T)[j,k]... no
            # v[j,k] * w[k,i]: sum over k? No, k is free.
            # Let's think: for fixed i, T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # = u[i,j] * (v * w.T)[j, k] where (v*w.T) has element (j,k) = sum_l v[j,l]*w[l,k]? No.
            # v[j,k] * w[k,i] is element-wise product of v (m x p) and w.T (p x n) but with index k shared.
            # Actually, define M = v @ w  -- no, that sums over k.
            # We want element (j,k) of something to be v[j,k] * w[k,i] for fixed i.
            # This is v * w[:,i].T? w[:,i] is a vector of length p.
            # So for fixed i, the matrix M_i[j,k] = v[j,k] * w[k,i] = v * w[:,i].T (element-wise)
            # Then T[i,j,k] += u[i,j] * M_i[j,k]
            # So T[i,:,:] += u[i,:] @ M_i  -- no, M_i is (m,p), u[i,:] is (m,)
            # T[i,:,:] += outer(u[i,:], ...)? No.
            # T[i,j,k] = u[i,j] * M_i[j,k]
            # This is: T[i] = u[i] ⊙ M_i (element-wise) but u[i] is (m,) and M_i is (m,p)
            # So T[i] = M_i * u[i][:, None]
            
            for i in range(n):
                # M_i[j,k] = v[j,k] * w[k,i]
                M_i = v * w[:, i][None, :]  # (m, p)
                T[i, :, :] += u[i, :][:, None] * M_i  # (m, p)
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    def verify_fast(U, V, W, rank):
        """Faster verification using matrix operations."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Let's compute this differently.
        # For each pair (i,k), define S[i,k,j] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Note W3[k,i,r] is scalar for fixed i,k,r.
        # S[i,k,j] = sum_r (U3[i,j,r] * V3[j,k,r]) * W3[k,i,r]
        # Let A_r[i,j,k] = U3[i,j,r] * V3[j,k,r] -- this is (n,m,p) for each r
        # Then S[i,k,j] = sum_r A_r[i,j,k] * W3[k,i,r]
        
        # Alternative approach: 
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # = sum_r U3[i,j,r] * (V3[j,k,r] * W3[k,i,r])
        
        # Let's just do the loop but vectorize over r
        T = np.zeros((n, m, p))
        for i in range(n):
            for k in range(p):
                # For fixed i,k, T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                # = sum_j' ... no
                # Let's compute for all j at once
                # U3[i,:,r] is (m,), V3[:,k,r] is (m,) -- wait, V3[j,k,r]
                # For fixed i,k: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                # = sum_r (U3[i,:,r] * V3[:,k,r]) * W3[k,i,r]  -- but this sums over r, and j is free
                # Actually: T[i,j,k] = sum_r [U3[i,j,r] * V3[j,k,r] * W3[k,i,r]]
                # For fixed i,k, this is a vector over j:
                # T[i,:,k] = sum_r [U3[i,:,r] * V3[:,k,r] * W3[k,i,r]]
                # = sum_r W3[k,i,r] * (U3[i,:,r] * V3[:,k,r])
                
                # U3[i,:,r] has shape (m,), V3[:,k,r] has shape (m,)
                # So U3[i,:,r] * V3[:,k,r] is element-wise product, shape (m,)
                # Then multiply by scalar W3[k,i,r] and sum over r
                
                # Vectorize over r:
                # U3[i, :, :] has shape (m, rank)
                # V3[:, k, :] has shape (m, rank)
                # Element-wise product: (m, rank)
                # Multiply by W3[k, i, :] (shape (rank,)) along rank axis
                # Sum over rank axis
                
                prod = U3[i, :, :] * V3[:, k, :]  # (m, rank)
                T[i, :, k] = prod @ W3[k, i, :]  # (m,) = sum_r prod[j,r] * W3[k,i,r]
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    # But realistic lower bound for matmul is higher.
    # Known results: 3x3x3 has rank 23 (Strassen), 4x4x4 has rank 49
    # For 3x5x7, we don't have known optimal, but let's try to improve on 105.
    
    np.random.seed(42)
    
    # Strategy: try random initializations and use gradient-free optimization
    # or structured random search
    
    # First, try some structured approaches
    # Approach 1: Try to use the fact that for small n, we can sometimes
    # combine terms using polynomial interpolation (Winograd-style)
    
    # For n=3, we can use 3-point interpolation: 
    # f(x) = a*x^2 + b*x + c, evaluate at x=0,1,2
    # This gives a rank-3 decomposition for the "diagonal" part
    
    # But this is complex. Let's try random search with verification.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Start from lower ranks and work up, or from schoolbook and work down?
    # Working down is better: start from schoolbook and try to reduce.
    
    # Actually, let's try a different approach: random initialization + iterative refinement
    # This is like a "rank reduction" algorithm.
    
    # For each target rank r, initialize U, V, W randomly, then try to
    # minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
    
    # This is a non-convex problem. Let's try a few random restarts.
    
    def random_decomposition(rank, seed=None):
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        return U, V, W
    
    def compute_tensor(U, V, W, rank):
        """Compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for k in range(p):
                prod = U3[i, :, :] * V3[:, k, :]  # (m, rank)
                T[i, :, k] = prod @ W3[k, i, :]  # (m,)
        return T
    
    # Try to find a decomposition of rank r by random search
    # We'll try several random initializations and check if any is close
    
    # Given time constraints, let's try a few target ranks
    # and a few random restarts for each
    
    target_ranks = [105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 38, 36, 35]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few random initializations
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U, V, W = random_decomposition(target_rank, seed=42 + restart * 1000)
            
            # Quick check: compute tensor and see if it's close to all-ones
            T = compute_tensor(U, V, W, target_rank)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-9:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            elif error < 0.1:
                # Close, but not exact. Try to refine?
                # For now, skip
                pass
    
    # If we found a better rank, convert to list format
    # The interface expects lists, not numpy arrays
    # But actually, let's check: the seed program returns lists
    # Let's return numpy arrays and see if that works, or convert to lists
    
    # Convert to lists for safety
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
