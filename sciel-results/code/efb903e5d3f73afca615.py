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
    
    # Verify schoolbook is correct
    S_school = np.zeros_like(T)
    for r in range(rank_school):
        S_school += np.einsum('a,b,c->abc', U_school[:, r], V_school[:, r], W_school[:, r])
    if np.linalg.norm(T - S_school) > 1e-10:
        # Should not happen, but safety
        pass
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try to find lower rank decompositions
    # Lower bound is max(nm, mp, pn)
    min_rank = max(nm, mp, pn)
    
    for R in range(min_rank, rank_school):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for init_idx in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(init_idx * 1000 + R)
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            # ALS iterations
            S = np.zeros_like(T)
            for r in range(R):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            prev_error = float('inf')
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                
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
                if error < 1e-8:
                    converged = True
                    break
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
            
            if not converged:
                continue
            
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            S_snap = np.zeros_like(T)
            for r in range(R):
                S_snap += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            snap_error = np.linalg.norm(T - S_snap)
            
            if snap_error < 1e-10:
                # Valid exact solution
                best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                break
        
        if best_rank == R:
            break
    
    # Convert to list of lists for interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
