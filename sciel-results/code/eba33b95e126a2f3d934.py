import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90

    # Build schoolbook decomposition
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

    # Build the target tensor T as a 3D array: T[w, u, v]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[w, u, v] = 1.0

    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    time_limit = 115  # Leave 5s margin for 120s total

    # We'll try ranks from 90 down, attempting to find a valid decomposition
    # Strategy: Use the structure of matrix multiplication to find rank reductions
    # 
    # Key insight: The matrix multiplication tensor T(i,j,k -> i,j,k) has the property
    # that T[i*m+j, j*p+k] = delta(i,j,k valid) which is 1 if the indices are consistent
    # (i.e., the middle index j matches). This is a "structured" tensor that may have
    # lower rank than the naive n*m*p.
    #
    # Approach: Try to find a rank-R decomposition by:
    # 1. Randomly initialize U (n*m x R) and V (m*p x R) with small integers
    # 2. For each w-slice, solve for W_w using least squares on the linear system
    #    T_w = sum_r W_wr * (U_r outer V_r)
    # 3. Check if the reconstruction is exact
    # 4. If exact, round to half-integers and verify

    # First, let's try a smarter initialization based on the structure
    # The schoolbook uses columns where only one (i,j,k) triple is active.
    # We can try to merge some of these using algebraic identities.
    
    # Try various ranks
    for R in range(rank_schoolbook - 1, 60, -1):
        if time.time() - start_time > time_limit:
            break

        found = False
        # Multiple random attempts
        max_attempts = 3 if R > 80 else 10
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(hash((R, attempt)) % (2**31))
            
            # Initialize U and V
            # Start from schoolbook first R columns and add structured perturbation
            if R <= rank_schoolbook:
                U_curr = U_sb[:, :R].copy().astype(float)
                V_curr = V_sb[:, :R].copy().astype(float)
            else:
                U_curr = np.zeros((n * m, R))
                V_curr = np.zeros((m * p, R))

            # Add random perturbation to break symmetry
            scale = 0.5 + attempt * 0.3
            U_curr += np.random.randn(n * m, R) * scale
            V_curr += np.random.randn(m * p, R) * scale

            # For each w-slice, build the linear system and solve for W_w
            # T_w = sum_r W_wr * outer(U_r, V_r)
            # vec(T_w) = sum_r W_wr * vec(outer(U_r, V_r))
            # Let M = [vec(outer(U_0, V_0)), vec(outer(U_1, V_1)), ..., vec(outer(U_{R-1}, V_{R-1}))]
            # Then vec(T_w) = M @ W_w (where W_w is R x 1)

            M = np.zeros((n * m * m * p, R))
            for r in range(R):
                M[:, r] = np.outer(U_curr[:, r], V_curr[:, r]).flatten()

            W_curr = np.zeros((p * n, R))
            total_residual = 0.0

            for w in range(p * n):
                t_vec = T[w].flatten()
                # Solve M @ W_w = t_vec
                W_w, residuals, rank_M, s = np.linalg.lstsq(M, t_vec, rcond=None)
                W_curr[w] = W_w
                if len(residuals) > 0:
                    total_residual += residuals[0]
                else:
                    total_residual += np.linalg.norm(M @ W_w - t_vec) ** 2

            # Check if the fit is good enough
            if total_residual < 1e-8:
                # Great! We have an approximate exact solution.
                # Now round to half-integers and verify
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2

                # Verify exactness with rounded values
                max_diff = 0.0
                for w in range(p * n):
                    recon_w = np.zeros((n * m, m * p))
                    for r in range(R):
                        if W_rounded[w, r] != 0:
                            recon_w += W_rounded[w, r] * np.outer(U_rounded[:, r], V_rounded[:, r])
                    diff = np.abs(recon_w - T[w])
                    max_diff = max(max_diff, np.max(diff))

                if max_diff < 1e-9:
                    best_rank = R
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break

        if found:
            # Try to go lower, but with less time per rank
            continue

    # If no better than schoolbook was found, return schoolbook
    # But we should also try a more systematic approach for the best rank found
    
    # Let's also try to verify and potentially improve using integer programming ideas
    # For now, return the best we found

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
