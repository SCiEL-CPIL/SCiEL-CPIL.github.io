import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb
    
    deadline = time.time() + 140  # Safety margin (150s limit)
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        M = np.einsum('ar,cr->acr', U, W)
        T_approx = np.einsum('acr,br->abc', M, V)
        return np.allclose(T_approx, T, atol=tol, rtol=0)
    
    def try_round_and_verify(U, V, W):
        """Try rounding to half-integers and verify"""
        for scale in [2, 4, 8, 16, 32, 64]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None
    
    def exterior_algebra_lower_bound():
        """
        Compute a lower bound on tensor rank using exterior algebra.
        For the matrix multiplication tensor M_{n,m,p}, we consider
        the exterior square of the input space and check the rank
        of the induced map.
        
        For 2x5x5: input space is R^{10} (A matrix), we look at
        how the tensor acts on wedge products.
        """
        # The tensor T can be seen as a bilinear map:
        # A (2x5) x B (5x5) -> C (2x5)
        # Flatten: A in R^{10}, B in R^{25}, C in R^{10}
        
        # For exterior algebra lower bound, we consider:
        # For each rank-1 term u_r ⊗ v_r ⊗ w_r, 
        # the contribution to the exterior square is limited.
        
        # A simpler approach: use the slice rank bound.
        # For each fixed c, T[:,:,c] is a matrix. The rank of
        # the decomposition is at least max_c rank(T[:,:,c]).
        
        # T[:,:,c] for c = k*n+i:
        # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k
        # For fixed c = k*n+i: a = i*m+j, b = j*p+k for j=0..4
        # This is a 10x25 matrix with 5 non-zero entries (one per j)
        # Each row a=i*m+j has non-zero at column b=j*p+k
        # So T[:,:,c] has exactly 5 ones, one in each "block row"
        # The rank of this matrix: rows are e_{i*m+j} and cols are e_{j*p+k}
        # These are 5 independent rows and 5 independent cols, so rank = 5
        
        # Similarly for other slices
        # T[:,b,:] for fixed b=j*p+k: non-zero at a=i*m+j, c=k*n+i
        # 2 non-zero entries, rank at most 2
        
        # T[a,:,:] for fixed a=i*m+j: non-zero at b=j*p+k, c=k*n+i for k=0..4
        # 5 non-zero entries, rank at most 2 (since only 2 distinct c values... wait)
        # c = k*n+i for k=0..4, so c takes values i, 5+i, 10+i, 15+i, 20+i... no
        # c = k*n+i = 2k+i for k=0..4, so c in {i, i+2, i+4, i+6, i+8}
        # All distinct, so rank is at most min(5,5) but the structure matters
        
        # The slice rank bound says: rank(T) >= max over all slices of rank(slice)
        # But this is a weak bound. For our tensor, max slice rank is 5.
        
        # A stronger bound: use the fact that for any decomposition
        # T = sum_r u_r ⊗ v_r ⊗ w_r, we have for each c:
        # T[:,:,c] = sum_r (u_r * w_r[c]) ⊗ v_r
        # So rank(T[:,:,c]) <= rank_r for all c
        # This gives rank >= max_c rank(T[:,:,c]) = 5 (weak)
        
        # Better: consider the "fiber" structure
        # For the matrix multiplication tensor, it's known that
        # the rank of M_{n,m,p} is at least n*m*p / max(n,m,p) in some cases
        # but this isn't always tight.
        
        # For our specific case, let's just return a reasonable lower bound
        # The true minimum is likely 50 (schoolbook) for 2x5x5
        # but let's compute a few bounds
        
        # Bound 1: slice rank
        max_slice_rank = 0
        for c in range(pn):
            mat = T[:, :, c]
            r = np.linalg.matrix_rank(mat, tol=1e-10)
            max_slice_rank = max(max_slice_rank, r)
        for b in range(mp):
            mat = T[:, b, :]
            r = np.linalg.matrix_rank(mat, tol=1e-10)
            max_slice_rank = max(max_slice_rank, r)
        for a in range(nm):
            mat = T[a, :, :]
            r = np.linalg.matrix_rank(mat, tol=1e-10)
            max_slice_rank = max(max_slice_rank, r)
        
        return max_slice_rank
    
    # Compute lower bound
    lb = exterior_algebra_lower_bound()
    
    # The lower bound is weak (likely 5), so we search from rank_sb-1 down
    # but with a smarter approach
    
    def try_rank(target_rank, max_iters=50):
        """
        Try to find a rank-target_rank decomposition using
        iterative refinement with random restarts.
        """
        for restart in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(12345 + restart)
            
            # Initialize with a mix of schoolbook and random
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            if target_rank <= rank_sb:
                # Start from a subset of schoolbook
                idx = np.random.choice(rank_sb, size=target_rank, replace=False)
                U = U_sb[:, idx].copy()
                V = V_sb[:, idx].copy()
                W = W_sb[:, idx].copy()
            
            # Add small random perturbation to escape local minima
            U += np.random.randn(nm, target_rank) * 0.1
            V += np.random.randn(mp, target_rank) * 0.1
            W += np.random.randn(pn, target_rank) * 0.1
            
            # Iterative refinement
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    return None
                
                M = np.einsum('ar,cr->acr', U, W)
                T_approx = np.einsum('acr,br->abc', M, V)
                R = T - T_approx
                res = np.sum(R**2)
                
                if res < 1e-12:
                    break
                
                # Compute gradients
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm < 1e-15:
                    break
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.999
            
            # Try to round to rational values
            rounded = try_round_and_verify(U, V, W)
            if rounded is not None:
                return rounded
            
            if verify(U, V, W, tol=1e-9):
                return U, V, W
        
        return None
    
    # Search for lower ranks
    # Focus on ranks that are "natural" reductions
    candidate_ranks = []
    for r in range(rank_sb - 1, max(lb, 24), -1):
        candidate_ranks.append(r)
    
    # Prioritize ranks that are multiples of small numbers
    prioritized = []
    for r in candidate_ranks:
        # Prefer ranks divisible by 2, 5, or 10
        score = 0
        for d in [2, 5, 10]:
            if r % d == 0:
                score += 1
        prioritized.append((score, r))
    prioritized.sort(key=lambda x: (-x[0], x[1]))
    
    # Try a subset of candidate ranks
    tried = set()
    for _, r in prioritized:
        if time.time() > deadline - 20:
            break
        if r in tried:
            continue
        if r >= best_rank:
            continue
        tried.add(r)
        
        result = try_rank(r, max_iters=10)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = r
                # Continue searching for even lower ranks
                continue
    
    return best_result
