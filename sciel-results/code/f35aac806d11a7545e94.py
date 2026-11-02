import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
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
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Start a timer
    start_time = time.time()
    time_limit = 110.0  # Keep a safety margin from 120s

    # Lower bound for rank
    # The rank is at least max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try to find a lower rank
    # We will iterate ranks from min_rank up to rank_schoolbook - 1
    # For each rank, we try a few random initializations and use ALS
    num_restarts = 10
    
    for R in range(min_rank, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly
            # Scale matters, but ALS usually handles scale by distributing it
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS iterations
            # We solve for one factor at a time by flattening the others
            # T is (A, B, C). U is (A, R), V is (B, R), W is (C, R).
            # Residual: T - sum_r U[:,r] V[:,r] W[:,r]
            # Fix V, W: For each r, we have a vector V_r and W_r.
            # The term is U_r * (V_r * W_r^T) ? No.
            # The tensor is sum_r outer(U_r, V_r, W_r).
            # Let M_VW be the matrix such that (M_VW)_{ab, r} = sum_c ... no.
            
            # Let's reshape T to (n*m, m*p * p*n).
            # T_flat[a, b*p*n + c] = T[a, b, c]
            # The model is: sum_r U[a, r] * (V[b, r] * W[c, r])
            # This is not linear in U if V and W are fixed?
            # Wait, for a fixed r, the contribution to T[a, b, c] is U[a, r] V[b, r] W[c, r].
            # If we fix V and W, the coefficient for U[a, r] in T[a, b, c] is V[b, r] W[c, r].
            # This is a linear system for U.
            # Let's construct the system for U.
            # For each r, we want to find U[:, r] such that
            # sum_r U[:, r] * (V[:, r] * W[:, r]^T) approx T
            # Actually, it's easier to think of it as:
            # Let X_r = V[:, r] * W[:, r]^T. This is a matrix of size (m*p, p*n).
            # Then T = sum_r U[:, r] * X_r (where * is outer product in the first index and matrix in the other two? No.)
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
            # This can be written as T = sum_r U[:, r] \otimes (V[:, r] \otimes W[:, r])? No.
            # Let's flatten the last two indices of T: T_flat[a, idx_bc] = T[a, b, c] where idx_bc = b * (p*n) + c.
            # Let's flatten the last two indices of the term for rank r:
            # Term_r[a, idx_bc] = U[a, r] * V[b, r] * W[c, r].
            # Note that V[b, r] * W[c, r] is the (b, c) entry of the matrix M_r = V[:, r] W[:, r]^T.
            # So Term_r is the matrix product? No, it's U[:, r] multiplied by M_r element-wise? No.
            # Term_r[a, b, c] = U[a, r] * (M_r)_{b, c}.
            # So Term_r is a tensor formed by scaling the matrix M_r by the vector U[:, r] in the first dimension.
            # We can reshape Term_r to a matrix of size (n*m, m*p * p*n) by flattening the last two dims.
            # Let T_flat be (n*m, m*p * p*n).
            # Let M_r_flat be the flattened M_r, size (m*p * p*n,).
            # Then Term_r_flat = U[:, r] * M_r_flat^T.
            # So T_flat = sum_r U[:, r] * M_r_flat^T.
            # This is a linear least squares problem for U:
            # T_flat = U * M_flats^T, where M_flats is a matrix of size (R, m*p * p*n) with rows M_r_flat.
            
            # Let's implement this ALS step.
            
            T_flat = T.reshape(n * m, m * p * p * n)
            T_flat_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            T_flat_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            
            for it in range(50):
                # Update U
                M_flats = np.zeros((R, m * p * p * n))
                for r in range(R):
                    M_r = np.outer(V[:, r], W[:, r])
                    M_flats[r, :] = M_r.flatten()
                # Solve M_flats * U.T = T_flat.T
                U_new, _, _, _ = np.linalg.lstsq(M_flats, T_flat.T, rcond=None)
                U = U_new.T
                
                # Update V
                # T_flat[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Fix U, W.
                # For each r, M_r = U[:, r] * W[:, r]^T ? No.
                # Term_r[b, a, c] = V[b, r] U[a, r] W[c, r].
                # Let's flatten a and c.
                # T_flat_ac[b, a, c] -> reshape to (m*p, n*m * p*n).
                # T_flat_ac = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                M_flats_V = np.zeros((R, n * m * p * n))
                for r in range(R):
                    M_r = np.outer(U[:, r], W[:, r])
                    M_flats_V[r, :] = M_r.flatten()
                V_new, _, _, _ = np.linalg.lstsq(M_flats_V, T_flat_V.T, rcond=None)
                V = V_new.T
                
                # Update W
                # T_flat_c[a, b, c] -> reshape to (p*n, n*m * m*p).
                # T_flat_c = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                M_flats_W = np.zeros((R, n * m * m * p))
                for r in range(R):
                    M_r = np.outer(U[:, r], V[:, r])
                    M_flats_W[r, :] = M_r.flatten()
                W_new, _, _, _ = np.linalg.lstsq(M_flats_W, T_flat_W.T, rcond=None)
                W = W_new.T
                
                # Check residual
                # Reconstruct T_est
                # T_est = sum_r outer(U[:, r], V[:, r], W[:, r])
                # We can compute this efficiently
                T_est = np.einsum('ar,br,cr->abc', U, V, W)
                res = np.linalg.norm(T - T_est)
                if res < 1e-6:
                    # Found a solution? Check if it can be rounded to integers/half-integers
                    # The problem states "integer/half-integer entries".
                    # This implies 2*entry is an integer.
                    # Let's check if rounding 2*U, 2*V, 2*W to nearest integer gives a valid solution.
                    U_rounded = np.round(2 * U) / 2.0
                    V_rounded = np.round(2 * V) / 2.0
                    W_rounded = np.round(2 * W) / 2.0
                    
                    T_est_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    res_rounded = np.linalg.norm(T - T_est_rounded)
                    
                    if res_rounded < 1e-6:
                        # Valid solution found with rank R
                        # Convert to lists of lists
                        best_rank = R
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        # We can break early if we find a low rank
                        # But maybe we can find even lower?
                        # Let's break the restart loop and continue to lower R?
                        # No, we are iterating R from low to high.
                        # So if we find one at R, it's the minimal rank in this search order?
                        # Not necessarily, but it's a valid lower rank.
                        # We can stop searching higher R.
                        # But we might find an even lower R in the next iteration of the outer loop?
                        # The outer loop is increasing R. So if we find R=25, we continue to R=26.
                        # We should stop if we find a solution, because we are going upwards.
                        # Wait, the loop is `for R in range(min_rank, rank_schoolbook)`.
                        # So if we find a solution at R, it is the smallest R found so far.
                        # We can break the outer loop?
                        # Yes, because we want the minimal rank.
                        break
            if best_rank == R:
                break

    # If no lower rank found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}

