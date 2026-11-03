import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    # Initialize with schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Build the target tensor slices
    # T_w is a matrix of size (n*m) x (m*p)
    # T_w[u, v] = 1 if u, v, w correspond to a valid multiplication
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T_slices = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_slices[w, u, v] = 1

    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()

    # Try to reduce rank
    # We try ranks from 89 down to 70
    # Time limit 120s.
    start_time = time.time()
    time_limit = 110  # Leave 10s margin

    for R in range(rank_schoolbook - 1, max(50, rank_schoolbook - 20), -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        found = False
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V using first R columns of schoolbook
            U_curr = U_sb[:, :R].copy()
            V_curr = V_sb[:, :R].copy()
            
            # Add some noise to escape local minima
            if attempt > 0:
                noise = 0.1
                U_curr += np.random.randn(n * m, R) * noise
                V_curr += np.random.randn(m * p, R) * noise
                
            # Initialize W
            W_curr = np.zeros((p * n, R))
            
            # Iterative optimization
            # We minimize sum_w || T_w - sum_r W_wr U_r V_r^T ||^2
            # This is equivalent to minimizing || T - sum_r w_r u_r v_r^T ||_F^2
            # where T is the 3D tensor.
            
            # We can use a simple alternating least squares (ALS)
            # But W is linear, U and V are bilinear.
            
            # Let's try a simple gradient step or just ALS on W, then refine U, V.
            # Since U, V are hard to update optimally, we might just fix U, V from schoolbook
            # and see if W can fit.
            # If U, V are from schoolbook, the span of {U_r V_r^T} is the same as schoolbook.
            # So W can fit exactly if R is sufficient to span the slices.
            # But we need the *same* U, V for all slices.
            # The schoolbook U, V are orthogonal-ish.
            # If we take a subset, we lose rank.
            
            # Let's try to solve for W given U, V.
            # For each w, T_w = sum_r W_wr (U_r V_r^T)
            # Let M_r = U_r V_r^T.
            # T_w = sum_r W_wr M_r
            # This is a linear system for W_wr.
            # Vectorize: vec(T_w) = sum_r W_wr vec(M_r)
            # Let A be a matrix with columns vec(M_r).
            # Then vec(T_w) = A W_w
            # We can solve for W_w using least squares.
            
            A = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_r = np.outer(U_curr[:, r], V_curr[:, r])
                A[:, r] = M_r.flatten()
                
            # Solve for W
            W_solution = np.zeros((p * n, R))
            residual_sum = 0
            for w in range(p * n):
                t_vec = T_slices[w].flatten()
                # Solve A W_w = t_vec
                # Use lstsq
                W_w, residuals, rank_A, s = np.linalg.lstsq(A, t_vec, rcond=None)
                W_solution[w] = W_w
                residual_sum += np.linalg.norm(A @ W_w - t_vec)
                
            # Check if residual is close to 0
            if residual_sum < 1e-6:
                # We have a solution with floating point numbers.
                # Now we need to round to half-integers.
                # This is a constraint satisfaction problem.
                # We check if rounding U, V, W yields an exact solution.
                
                # Since U, V were initialized from schoolbook (integers),
                # and we only added small noise or kept them,
                # the optimal W might be close to integers?
                # Actually, if U, V are integers, and T is integers, W might be rational.
                
                # Let's try rounding U, V, W to nearest half-integers.
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_solution * 2) / 2
                
                # Verify exactness
                # Reconstruct T
                T_recon = np.zeros((p * n, n * m, m * p))
                for w in range(p * n):
                    for r in range(R):
                        if W_rounded[w, r] != 0:
                            T_recon[w] += W_rounded[w, r] * np.outer(U_rounded[:, r], V_rounded[:, r])
                
                # Check if T_recon matches T_slices
                # T_slices has 1s at specific positions, 0 elsewhere.
                # Due to floating point, we check difference.
                diff = np.abs(T_recon - T_slices)
                if np.max(diff) < 1e-9:
                    # Success!
                    best_rank = R
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    found = True
                    break
            
            # If we found a solution, break attempt loop
            if found:
                break
        
        # If we found a solution for this R, we can stop (since we want minimal rank)
        # But we are iterating downwards, so the first found is the best so far.
        # We can continue to see if we can go lower, but time is limited.
        # Let's stop at the first found to be safe and fast.
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
