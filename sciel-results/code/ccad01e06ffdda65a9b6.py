"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute T reshapes for fast ALS
    T_nm = T.reshape(nm, mp * pn)
    T_mp = T.reshape(mp, nm * pn)
    T_pn = T.reshape(pn, nm * mp)
    
    # Schoolbook decomposition
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-10):
        # Efficient check using einsum
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def compute_residual(U, V, W):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(S - T)
    
    def fast_als(U, V, W, max_iters=100, tol=1e-10):
        """
        Optimized ALS.
        """
        r = U.shape[1]
        if r == 0:
            return None
        
        # Copy to avoid modifying original
        Ur = U.copy()
        Vr = V.copy()
        Wr = W.copy()
        
        # Precompute outer products of pairs? No, too big.
        # Just loop.
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # S_total = sum_r U_r V_r W_r
            # We need S_minus_r = S_total - U_r V_r W_r
            # But S_total changes as we update U. 
            # Standard ALS updates one column at a time, updating S_total incrementally.
            
            # Initialize S_total
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            
            for r_idx in range(r):
                # Remove current term
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                
                # Solve for Ur[:, r_idx]
                # Minimize || S_minus - U_new * V_r * W_r ||
                # Reshape S_minus to (nm, mp*pn)
                S_minus_mat = S_minus.reshape(nm, mp * pn)
                vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
                vw_norm_sq = np.dot(vw, vw)
                
                if vw_norm_sq > 1e-15:
                    Ur_new = S_minus_mat @ vw / vw_norm_sq
                    # Update S_total
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur_new, Vr[:, r_idx], Wr[:, r_idx])
                    Ur[:, r_idx] = Ur_new
            
            # Update V
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                
                S_minus_mat = S_minus.reshape(mp, nm * pn)
                uw = np.outer(Ur[:, r_idx], Wr[:, r_idx]).flatten()
                uw_norm_sq = np.dot(uw, uw)
                
                if uw_norm_sq > 1e-15:
                    Vr_new = S_minus_mat @ uw / uw_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur[:, r_idx], Vr_new, Wr[:, r_idx])
                    Vr[:, r_idx] = Vr_new
            
            # Update W
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                
                S_minus_mat = S_minus.reshape(pn, nm * mp)
                uv = np.outer(Ur[:, r_idx], Vr[:, r_idx]).flatten()
                uv_norm_sq = np.dot(uv, uv)
                
                if uv_norm_sq > 1e-15:
                    Wr_new = S_minus_mat @ uv / uv_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr_new)
                    Wr[:, r_idx] = Wr_new
            
            # Check convergence
            if it % 5 == 0 or it == max_iters - 1:
                res = np.linalg.norm(S_total - T)
                if res < tol:
                    if verify(Ur, Vr, Wr):
                        return Ur, Vr, Wr
                        
        return None

    def try_rank(r, n_attempts=5, max_iters=200):
        """
        Try to find a valid rank-r decomposition.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Initialization
            if attempt == 0:
                # Random Gaussian
                scale = 1.0
                Ur = np.random.randn(nm, r) * scale
                Vr = np.random.randn(mp, r) * scale
                Wr = np.random.randn(pn, r) * scale
            elif attempt == 1:
                # Sparse binary
                Ur = (np.random.rand(nm, r) < 0.5).astype(float)
                Vr = (np.random.rand(mp, r) < 0.5).astype(float)
                Wr = (np.random.rand(pn, r) < 0.5).astype(float)
            else:
                # Mix of random and zero
                Ur = np.random.randn(nm, r)
                Vr = np.random.randn(mp, r)
                Wr = np.random.randn(pn, r)
                # Mask some to zero
                Ur[np.random.rand(nm, r) < 0.5] = 0
                Vr[np.random.rand(mp, r) < 0.5] = 0
                Wr[np.random.rand(pn, r) < 0.5] = 0
            
            result = fast_als(Ur, Vr, Wr, max_iters=max_iters)
            if result is not None:
                return result
        
        return None

    # Search from lower bound upwards
    # Lower bound is max(nm, mp, pn) = 28
    lower_bound = max(nm, mp, pn)
    
    # We might not find rank 28. We should try a few ranks.
    # To be safe, we iterate from lower_bound to best_rank.
    # But ALS is expensive. We limit attempts.
    
    # Strategy: Try ranks [28, 29, 30, 31, 32, ...]
    # If we find one, return it.
    # If we run out of time, return best_U (schoolbook).
    
    # To maximize chance of finding a low rank, we can prioritize lower ranks.
    # However, lower ranks are harder. 
    # Let's try 28, 29, 30 with a few attempts each.
    
    search_ranks = list(range(lower_bound, min(lower_bound + 10, best_rank)))
    
    for target_rank in search_ranks:
        if time.time() > deadline:
            break
            
        # Try this rank
        res = try_rank(target_rank, n_attempts=3, max_iters=100)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
            # Since we are searching upwards from the minimum, 
            # the first valid one is the minimum found in this sequence.
            # We can break if we found a very good one? 
            # No, we want the *minimum*. So if we found rank 30, 
            # we should stop? 
            # Wait, if we search 28, 29, 30... and find 30, 
            # it's possible 29 exists but we missed it.
            # However, if we found 28, we should stop immediately.
            if target_rank == lower_bound:
                break
            # Otherwise, continue to see if we can find lower? 
            # No, we are iterating UP. So if we found 30, we have already passed 28, 29.
            # So 30 is the best we found so far in this ascending order.
            # We should break because any higher rank is worse.
            break

    # If we didn't find anything better, we might try a few more random high-rank searches?
    # No, schoolbook is a valid fallback.
    
    # Convert to lists
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
