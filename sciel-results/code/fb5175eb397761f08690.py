"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    best_U, best_V, best_W = U_s, V_s, W_s

    deadline = time.time() + 130  # safety margin before 150s kill

    def verify(U, V, W, tol=1e-10):
        # Efficient check using einsum
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    def compute_residual(U, V, W):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(S - T)

    def fast_als(U, V, W, max_iters=200, tol=1e-12):
        """
        Optimized ALS with convergence check.
        """
        r = U.shape[1]
        if r == 0:
            return None
        
        Ur = U.copy()
        Vr = V.copy()
        Wr = W.copy()
        
        # Precompute outer products for efficiency
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(nm, mp * pn)
                vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
                vw_norm_sq = np.dot(vw, vw)
                if vw_norm_sq > 1e-15:
                    Ur_new = S_minus_mat @ vw / vw_norm_sq
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
            if it % 3 == 0 or it == max_iters - 1:
                res = np.linalg.norm(S_total - T)
                if res < tol:
                    if verify(Ur, Vr, Wr):
                        return Ur, Vr, Wr
                        
        return None

    def try_rank(r, n_attempts=10, max_iters=300):
        """
        Try to find a valid rank-r decomposition.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
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
            elif attempt == 2:
                # Mix of random and zero
                Ur = np.random.randn(nm, r)
                Vr = np.random.randn(mp, r)
                Wr = np.random.randn(pn, r)
                Ur[np.random.rand(nm, r) < 0.5] = 0
                Vr[np.random.rand(mp, r) < 0.5] = 0
                Wr[np.random.rand(pn, r) < 0.5] = 0
            elif attempt == 3:
                # Structured initialization based on schoolbook
                Ur = np.zeros((nm, r))
                Vr = np.zeros((mp, r))
                Wr = np.zeros((pn, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = (i * m + j) * p + k
                            if idx < r:
                                Ur[i*m+j, idx] = 1
                                Vr[j*p+k, idx] = 1
                                Wr[k*n+i, idx] = 1
                # Add some random perturbation
                Ur += 0.1 * np.random.randn(nm, r)
                Vr += 0.1 * np.random.randn(mp, r)
                Wr += 0.1 * np.random.randn(pn, r)
            elif attempt == 4:
                # Random with some structure
                Ur = np.zeros((nm, r))
                Vr = np.zeros((mp, r))
                Wr = np.zeros((pn, r))
                for idx in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    Ur[i*m+j, idx] = 1
                    Vr[j*p+k, idx] = 1
                    Wr[k*n+i, idx] = 1
                Ur += 0.1 * np.random.randn(nm, r)
                Vr += 0.1 * np.random.randn(mp, r)
                Wr += 0.1 * np.random.randn(pn, r)
            else:
                # Various random initializations
                Ur = np.random.randn(nm, r)
                Vr = np.random.randn(mp, r)
                Wr = np.random.randn(pn, r)
                # Randomly zero out some entries
                for mat in [Ur, Vr, Wr]:
                    mat[np.random.rand(*mat.shape) < 0.3] = 0
            
            result = fast_als(Ur, Vr, Wr, max_iters=max_iters)
            if result is not None:
                return result
        
        return None

    # Search from lower bound upwards
    # Lower bound is max(nm, mp, pn) = 28
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower_bound to best_rank - 1
    # We'll try a few ranks and see if we can find a valid decomposition
    search_ranks = list(range(lower_bound, min(lower_bound + 8, best_rank)))
    
    for target_rank in search_ranks:
        if time.time() > deadline:
            break
            
        # Try this rank
        res = try_rank(target_rank, n_attempts=8, max_iters=200)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
            # Since we are searching upwards from the minimum, 
            # the first valid one is the minimum found in this sequence.
            # We can break if we found a very good one? 
            # No, we want the *minimum*. So if we found rank 30, 
            # it's possible 29 exists but we missed it.
            # However, if we found 28, we should stop immediately.
            if target_rank == lower_bound:
                break
            # Otherwise, continue to see if we can find lower? 
            # No, we are iterating UP. So if we found 30, we have already passed 28, 29.
            # So 30 is the best we found so far in this ascending order.
            # We should break because any higher rank is worse.
            break

    # Convert to lists
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
