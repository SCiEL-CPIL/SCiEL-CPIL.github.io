"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    Searches for rank-minimal exact decomposition.
    """
    n, m, p = DIMENSIONS
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 140
    
    def verify_np(U, V, W):
        """Fast verification using NumPy tensor contraction."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Compute the tensor: T[i,j,k,ip,jp,kp] = sum_r U[i*m+j,r]*V[jp*p+k,r]*W[kp*n+ip,r]
        # We need T[i,j,k,ip,jp,kp] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        # Reshape for batched computation
        U_reshaped = U.reshape(n, m, rank)  # [i, j, r]
        V_reshaped = V.reshape(m, p, rank)  # [jp, k, r]
        W_reshaped = W.reshape(p, n, rank)  # [kp, ip, r]
        
        # For each (i, j, ip, jp, kp, k), compute sum_r U[i,j,r]*V[jp,k,r]*W[kp,ip,r]
        # Build expected tensor
        expected = np.zeros((n, m, p, n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i, j, k, i, j, k] = 1.0
        
        # Compute actual tensor via einsum
        # T[i,j,k,ip,jp,kp] = sum_r U[i,j,r] * V[jp,k,r] * W[kp,ip,r]
        # Note: V index is (jp, k), W index is (kp, ip)
        actual = np.einsum('ijr,jkr,kpr->ijkikp', U_reshaped, V_reshaped, W_reshaped)
        # Wait, let me be more careful with indices
        # U[i,j,r], V[jp,k,r], W[kp,ip,r]
        # We want to sum over r to get T[i,j,k,ip,jp,kp]
        # Let me use explicit loops for the outer dims and einsum for r
        max_diff = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                val = np.dot(U[i, j, :], V[jp, k, :]) * W[kp, ip, :]
                                # That's not right either. Let me do it properly.
                                # sum_r U[i,j,r] * V[jp,k,r] * W[kp,ip,r]
                                val = np.sum(U[i, j, :] * V[jp, k, :] * W[kp, ip, :])
                                exp = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                diff = abs(val - exp)
                                if diff > max_diff:
                                    max_diff = diff
        return max_diff < 1e-9
    
    def verify_fast(U, V, W):
        """Even faster verification using matrix multiplication tricks."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        max_diff = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                val = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                                exp = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                diff = abs(val - exp)
                                if diff > max_diff:
                                    max_diff = diff
                                if max_diff > 1e-9:
                                    return False
        return True
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Precompute all required tensor entries
    # The tensor has 1s at positions (i,j,k,i,j,k) for i in [0,n), j in [0,m), k in [0,p)
    # All other entries are 0
    
    # Strategy: Try random rank-1 tensors and see if they can be combined
    # A rank-1 tensor is defined by vectors u (length n*m), v (length m*p), w (length p*n)
    # The tensor entry T[i,j,k,ip,jp,kp] = u[i*m+j]*v[jp*p+k]*w[kp*n+ip]
    
    # For the multiplication tensor, we need:
    # sum_r u_r[i*m+j] * v_r[jp*p+k] * w_r[kp*n+ip] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    # Try structured random search with small integer values
    random.seed(42)
    
    for rank in range(20, rank_sb):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random attempts at this rank
        max_attempts = 50 if rank < 40 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse factors with small integer entries
            U = [[random.choice([-1, 0, 0, 0, 1, 1, 2, -2]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 0, 0, 1, 1, 2, -2]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 0, 0, 1, 1, 2, -2]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify_fast(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        # Also try a more structured approach: start from schoolbook and merge
        if time.time() > deadline - 5:
            break
        
        if rank < best_rank:
            # Try to find a rank-rank decomposition by random projection
            # This is less likely to work but worth trying
            pass
    
    # Final: try a few more aggressive random searches at the best rank found
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution
            U = [row[:] for row in best_U]
            V = [row[:] for row in best_V]
            W = [row[:] for row in best_W]
            # Randomly zero out some entries and adjust
            for _ in range(best_rank // 2):
                ri = random.randint(0, len(U)-1)
                rj = random.randint(0, best_rank-1)
                old_val = U[ri][rj]
                U[ri][rj] = 0
                # Compensate in V or W
                if random.random() < 0.5:
                    rk = random.randint(0, len(V)-1)
                    V[rk][rj] += old_val
                else:
                    rk = random.randint(0, len(W)-1)
                    W[rk][rj] += old_val
            
            if verify_fast(U, V, W):
                # Try to reduce rank by removing zero columns
                nonzero_cols = set()
                for row in U:
                    for idx, val in enumerate(row):
                        if val != 0:
                            nonzero_cols.add(idx)
                for row in V:
                    for idx, val in enumerate(row):
                        if val != 0:
                            nonzero_cols.add(idx)
                for row in W:
                    for idx, val in enumerate(row):
                        if val != 0:
                            nonzero_cols.add(idx)
                
                new_rank = len(nonzero_cols)
                if new_rank < best_rank:
                    col_map = {old: new for new, old in enumerate(sorted(nonzero_cols))}
                    U_new = [[U[i][col_map[j]] for j in range(new_rank)] for i in range(n*m)]
                    V_new = [[V[i][col_map[j]] for j in range(new_rank)] for i in range(m*p)]
                    W_new = [[W[i][col_map[j]] for j in range(new_rank)] for i in range(p*n)]
                    if verify_fast(U_new, V_new, W_new):
                        best_rank = new_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
