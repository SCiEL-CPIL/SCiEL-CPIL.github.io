"""Improved program for matrix multiplication tensor decomposition; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    # nm=12, mp=28, pn=21
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

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
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()

    deadline = time.time() + 145  # safety margin before 150s kill

    def verify(U, V, W, tol=1e-10):
        # Efficient check using einsum
        S = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(S, T, atol=tol)

    def fast_als(Ur, Vr, Wr, max_iters=200, tol=1e-10):
        """
        Optimized ALS with better convergence checking.
        """
        r = Ur.shape[1]
        if r == 0:
            return None
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr, optimize=True)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx], optimize=True)
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(nm, mp * pn)
                vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
                vw_norm_sq = np.dot(vw, vw)
                if vw_norm_sq > 1e-15:
                    Ur_new = S_minus_mat @ vw / vw_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur_new, Vr[:, r_idx], Wr[:, r_idx], optimize=True)
                    Ur[:, r_idx] = Ur_new
            
            # Update V
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr, optimize=True)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx], optimize=True)
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(mp, nm * pn)
                uw = np.outer(Ur[:, r_idx], Wr[:, r_idx]).flatten()
                uw_norm_sq = np.dot(uw, uw)
                if uw_norm_sq > 1e-15:
                    Vr_new = S_minus_mat @ uw / uw_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur[:, r_idx], Vr_new, Wr[:, r_idx], optimize=True)
                    Vr[:, r_idx] = Vr_new
            
            # Update W
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr, optimize=True)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx], optimize=True)
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(pn, nm * mp)
                uv = np.outer(Ur[:, r_idx], Vr[:, r_idx]).flatten()
                uv_norm_sq = np.dot(uv, uv)
                if uv_norm_sq > 1e-15:
                    Wr_new = S_minus_mat @ uv / uv_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr_new, optimize=True)
                    Wr[:, r_idx] = Wr_new
            
            # Check convergence
            if it % 3 == 0 or it == max_iters - 1:
                res = np.linalg.norm(S_total - T)
                if res < tol:
                    if verify(Ur, Vr, Wr):
                        return Ur, Vr, Wr
                        
        return None

    def try_rank(r, n_attempts=10, max_iters=150):
        """
        Try to find a valid rank-r decomposition with diverse initializations.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Random Gaussian
                Ur = np.random.randn(nm, r)
                Vr = np.random.randn(mp, r)
                Wr = np.random.randn(pn, r)
            elif attempt == 1:
                # Sparse binary
                Ur = (np.random.rand(nm, r) < 0.3).astype(float)
                Vr = (np.random.rand(mp, r) < 0.3).astype(float)
                Wr = (np.random.rand(pn, r) < 0.3).astype(float)
            elif attempt == 2:
                # Sparse with larger values
                Ur = (np.random.rand(nm, r) < 0.4).astype(float) * 2.0
                Vr = (np.random.rand(mp, r) < 0.4).astype(float) * 2.0
                Wr = (np.random.rand(pn, r) < 0.4).astype(float) * 2.0
            elif attempt == 3:
                # Sparse with alternating signs
                Ur = (np.random.rand(nm, r) < 0.5).astype(float) * (1 - 2*(np.random.rand(nm, r) < 0.5))
                Vr = (np.random.rand(mp, r) < 0.5).astype(float) * (1 - 2*(np.random.rand(mp, r) < 0.5))
                Wr = (np.random.rand(pn, r) < 0.5).astype(float) * (1 - 2*(np.random.rand(pn, r) < 0.5))
            elif attempt == 4:
                # Random with zeros
                Ur = np.random.randn(nm, r)
                Vr = np.random.randn(mp, r)
                Wr = np.random.randn(pn, r)
                mask = np.random.rand(nm, r) < 0.5
                Ur[mask] = 0
                mask = np.random.rand(mp, r) < 0.5
                Vr[mask] = 0
                mask = np.random.rand(pn, r) < 0.5
                Wr[mask] = 0
            else:
                # Pure random
                Ur = np.random.randn(nm, r)
                Vr = np.random.randn(mp, r)
                Wr = np.random.randn(pn, r)
            
            result = fast_als(Ur, Vr, Wr, max_iters=max_iters)
            if result is not None:
                return result
        
        return None

    # Search from lower bound upwards
    lower_bound = max(nm, mp, pn)
    
    # Try each rank from lower bound to best_rank-1
    for target_rank in range(lower_bound, min(lower_bound + 15, best_rank)):
        if time.time() > deadline:
            break
        
        # Try this rank
        res = try_rank(target_rank, n_attempts=8, max_iters=120)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
            # Found a valid decomposition; keep searching for lower? 
            # No, we're going UP. But we want the MINIMUM.
            # Since we're going from low to high, the first valid is the minimum in our search.
            # But we might have missed a lower one. However, lower_bound is the theoretical minimum.
            # If we found lower_bound, that's optimal.
            if target_rank == lower_bound:
                break
            # Continue to see if we can find a lower rank? No, we're going up.
            # But we should keep searching in case we missed the true minimum.
            # Actually, if we found rank 30, we've already tried 28 and 29.
            # So 30 is the best we found. Break.
            break

    # Convert to lists
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
