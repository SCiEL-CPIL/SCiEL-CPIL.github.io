"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = get_schoolbook()
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def verify_decomposition(U, V, W, R, tol=1e-10):
        """Check if U, V, W exactly decompose T"""
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(T - S) < tol
    
    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap
    
    def try_als_sparse(U, V, W, R, max_iters=50, tol=1e-10, deadline=110.0):
        """Run ALS iterations with sparse updates, return (U, V, W, converged)"""
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False
            
            # Update U
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                vw_mat = np.outer(V[:, r], W[:, r])
                numerator = np.einsum('ijk,jk->i', T - S, vw_mat)
                denominator = np.sum(vw_mat ** 2)
                if denominator > 1e-10:
                    U[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Update V
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                uw_mat = np.outer(U[:, r], W[:, r])
                numerator = np.einsum('ijk,ik->j', T - S, uw_mat)
                denominator = np.sum(uw_mat ** 2)
                if denominator > 1e-10:
                    V[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Update W
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                uv_mat = np.outer(U[:, r], V[:, r])
                numerator = np.einsum('ijk,ij->k', T - S, uv_mat)
                denominator = np.sum(uv_mat ** 2)
                if denominator > 1e-10:
                    W[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T - S)
            if error < tol:
                return U, V, W, True
            if abs(prev_error - error) < 1e-12:
                return U, V, W, False
            prev_error = error
        
        return U, V, W, False

    # Lower bound: max(nm, mp, pn)
    min_rank = max(nm, mp, pn)
    
    # Search for lower rank
    # We start from rank_school - 1 down to min_rank
    # But searching all ranks is expensive. We will try a few specific ranks.
    # Given the size, it's unlikely we find rank < 48 easily.
    # We will try ranks 47, 46, 45, 44, 43, 42, 41, 40.
    
    ranks_to_try = list(range(rank_school - 1, max(min_rank, rank_school - 10), -1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Multiple initialization strategies for this rank
        # We use sparse random initialization to encourage sparse solutions
        for init_idx in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(init_idx * 7919 + R * 131 + 42)
            
            # Sparse Random Initialization
            # Density 0.2
            density = 0.2
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            for r in range(R):
                # Pick random indices
                u_mask = np.random.rand(nm) < density
                v_mask = np.random.rand(mp) < density
                w_mask = np.random.rand(pn) < density
                
                # Ensure at least one non-zero
                if not np.any(u_mask): u_mask[np.random.randint(nm)] = True
                if not np.any(v_mask): v_mask[np.random.randint(mp)] = True
                if not np.any(w_mask): w_mask[np.random.randint(pn)] = True
                
                U[u_mask, r] = np.random.choice([-1, 1], size=np.sum(u_mask))
                V[v_mask, r] = np.random.choice([-1, 1], size=np.sum(v_mask))
                W[w_mask, r] = np.random.choice([-1, 1], size=np.sum(w_mask))
            
            U_out, V_out, W_out, converged = try_als_sparse(U, V, W, R, max_iters=30, deadline=deadline)
            
            if not converged:
                continue
            
            # Try snapping to half-integers
            U_snap, V_snap, W_snap = snap_to_half_integers(U_out, V_out, W_out)
            
            if verify_decomposition(U_snap, V_snap, W_snap, R):
                best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                break
        
        if best_rank == R:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
