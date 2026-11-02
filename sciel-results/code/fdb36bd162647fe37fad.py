"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 60) as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 145.0 # Safety margin

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Try to find lower rank using non-convex optimization
    # We know rank <= 60.
    # Let's try ranks from 55 down to 40.
    # For each rank, try a few random restarts.
    
    for r in range(55, 35, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for restart in range(20): # 20 restarts per rank
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W with random small integers
            U = np.random.randint(-2, 3, (n*m, r))
            V = np.random.randint(-2, 3, (m*p, r))
            W = np.random.randint(-2, 3, (p*n, r))
            
            # Local search / Gradient descent style
            converged = False
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                c = np.random.randint(r)
                
                # Compute residual
                # R = T - sum_{c' != c} U[:, c'] outer V[:, c'] outer W[:, c']
                # Efficient way:
                # Sum all, subtract column c
                # Tensor product sum: np.einsum('ic,jc,kc->ijk', U, V, W)
                # This is expensive for large r.
                # But r is small (< 60).
                # n*m=10, m*p=30, p*n=12. Size 10*30*12 = 3600.
                # np.einsum is fast.
                
                current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
                residual = T - current_sum + np.einsum('ic,jc,kc->ijk', U[:, c:c+1], V[:, c:c+1], W[:, c:c+1])
                
                # Find best rank-1 approx to residual
                # Use power method
                u = U[:, c].copy()
                v = V[:, c].copy()
                w = W[:, c].copy()
                
                for _ in range(5): # Few power iterations
                    # u_new = residual contracted with v, w
                    u_new = np.einsum('ijk,j,k->i', residual, v, w)
                    v_new = np.einsum('ijk,i,k->j', residual, u, w)
                    w_new = np.einsum('ijk,i,j->k', residual, u, v)
                    
                    norm_u = np.linalg.norm(u_new)
                    norm_v = np.linalg.norm(v_new)
                    norm_w = np.linalg.norm(w_new)
                    
                    if norm_u == 0 or norm_v == 0 or norm_w == 0:
                        break
                    
                    u_new /= norm_u
                    v_new /= norm_v
                    w_new /= norm_w
                    
                    u, v, w = u_new, v_new, w_new
                
                # Scale to match magnitude?
                # The power method finds direction. We need to scale.
                # We want u o v o w to approximate residual.
                # Let u = alpha * u_dir, etc.
                # Product alpha*beta*gamma = scale.
                # We can set alpha = beta = gamma = 1/scale^(1/3)?
                # Or just use the direction and find best scale?
                # Actually, we can just replace U[:, c] with u, etc, and let the next iteration handle scaling?
                # No, we need to update the actual values.
                
                # Let's try to find optimal scaling.
                # Minimize || residual - s * (u o v o w) ||^2
                # s = <residual, u o v o w> / ||u o v o w||^2
                uo = np.einsum('i,j,k->ijk', u, v, w)
                denom = np.sum(uo * uo)
                if denom == 0:
                    continue
                s = np.sum(residual * uo) / denom
                
                U[:, c] = s * u
                V[:, c] = s * v
                W[:, c] = s * w
                # Note: We distributed s equally? No, we put all s in U?
                # If we put s in U, then V and W are unit vectors.
                # This is a valid representation.
                
                # Check convergence
                current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
                if np.linalg.norm(T - current_sum) < 1e-9:
                    converged = True
                    break
            
            if converged:
                # We have a real-valued solution.
                # Try to round to half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    # Convert to lists
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    # We found a better rank, break out of restart loop?
                    # Yes, lower rank is better.
                    found = True
                    break
        
        # If we found a rank, we can stop searching lower ranks?
        # No, we want the MINIMAL rank.
        # But if we found r=50, we should check 49, 48...
        # However, if we break the restart loop, we continue to the next r (which is lower).
        # So the logic is correct.

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
