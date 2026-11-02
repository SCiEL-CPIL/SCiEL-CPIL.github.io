import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    schoolbook = get_schoolbook()
    best_rank = n * m * p
    best_solution = schoolbook

    start_time = time.time()
    time_limit = 140  # Leave 10s buffer for 150s limit

    def compute_error(U, V, W, R):
        """Compute the error of the decomposition.
        T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1 for all (i,j,k).
        """
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (s - 1.0) ** 2
        return error

    def check_half_integer(mat, tol=1e-4):
        """Check if all entries are close to half-integers."""
        for row in mat:
            for val in row:
                closest_half = round(val * 2) / 2
                if abs(val - closest_half) > tol:
                    return False
        return True

    def snap_to_half_integer(mat):
        """Snap values to nearest half-integer."""
        return np.round(mat * 2) / 2

    def als_solve(U, V, W, R, max_iters=100, tol=1e-8):
        """Run ALS iterations to converge.
        For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
        """
        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break

            # Update U: for each (i,j), minimize over U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # We need sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is linear in U[u_idx,:]: for each k, sum_c U[u_idx,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
                    A = np.zeros((p, R))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[u_idx, :] = sol
                    except:
                        pass

            # Update V: for each (j,k), minimize over V[j*p+k,:]
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    # sum_c U[i*m+j,c] * V[v_idx,c] * W[k*n+i,c] = 1 for all i
                    A = np.zeros((n, R))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[v_idx, :] = sol
                    except:
                        pass

            # Update W: for each (k,i), minimize over W[k*n+i,:]
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_idx,c] = 1 for all j
                    A = np.zeros((m, R))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[w_idx, :] = sol
                    except:
                        pass

            error = compute_error(U, V, W, R)
            if abs(prev_error - error) < tol:
                break
            prev_error = error

        return error

    def try_polish_and_verify(U, V, W, R):
        """Polish with ALS, snap to half-integers, verify. Return (rank, solution) or None."""
        if time.time() - start_time > time_limit:
            return None

        U = U.copy()
        V = V.copy()
        W = W.copy()

        # Run ALS to converge
        error = als_solve(U, V, W, R, max_iters=150, tol=1e-10)

        if error > 1e-4:
            return None

        # Try snapping to half-integers
        if check_half_integer(U, tol=1e-3) and check_half_integer(V, tol=1e-3) and check_half_integer(W, tol=1e-3):
            U_snapped = snap_to_half_integer(U)
            V_snapped = snap_to_half_integer(V)
            W_snapped = snap_to_half_integer(W)
            verify_error = compute_error(U_snapped, V_snapped, W_snapped, R)
            if verify_error < 1e-6:
                return (R, {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()})

        # Try integer snapping
        if check_half_integer(U, tol=1e-3):
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            verify_error = compute_error(U_int, V_int, W_int, R)
            if verify_error < 1e-6:
                return (R, {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()})

        return None

    # ============================================================
    # Strategy 1: Structured block decomposition
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x5) = [B1 (2x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    # ============================================================

    # For A1*B1 (2x2 * 2x5), use Strassen-style rank-7 for the 2x2 part
    # extended to 5 columns. Actually, let's think more carefully.
    #
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Split: C = C_block + C_tail
    # C_block[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C_block[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # C_tail[0,k] = A[0,2]*B[2,k]
    # C_tail[1,k] = A[1,2]*B[2,k]
    #
    # C_tail needs 2*5 = 10 scalar multiplications (schoolbook).
    # C_block is 2x2 * 2x5. We can use Strassen for the 2x2 part.
    # Strassen for 2x2: 7 multiplications of 2x2 blocks.
    # But here B is 2x5, so each "block multiplication" is 2x5.
    # Actually, Strassen computes M1 = (A11+A22)(B11+B22), etc.
    # For 2x2 * 2x5:
    # M1 = (A[0,0]+A[1,1]) * (B[0,:] + B[1,:])  -> 2x5 result, 10 scalar mults
    # M2 = A[1,:] * (B[0,:] + B[1,:])           -> 2x5 result, 10 scalar mults
    # M3 = A[0,:] * B[0,:]                       -> 2x5 result, 10 scalar mults
    # M4 = A[1,:] * B[1,:]                       -> 2x5 result, 10 scalar mults
    # M5 = (A[0,:]+A[1,:]) * B[1,:]              -> 2x5 result, 10 scalar mults
    # M6 = (A[0,0]-A[1,0]) * (B[1,0]+B[1,1])     -- this doesn't work for 2x5
    #
    # Actually, Strassen doesn't directly extend this way. Let me use a different approach.
    #
    # For 2x2 * 2x5, the schoolbook rank is 4*5 = 20.
    # We can try to find a lower rank decomposition.
    #
    # Let me try a more direct approach: parameterize the full tensor and use ALS.

    # ============================================================
    # Strategy 2: Random restarts with ALS, trying various ranks
    # ============================================================

    np.random.seed(42)

    for R in range(29, 24, -1):
        if time.time() - start_time > time_limit:
            break

        found = False
        num_attempts = 40

        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break

            # Different initialization strategies
            if attempt % 4 == 0:
                U = np.random.randn(dim_u, R)
                V = np.random.randn(dim_v, R)
                W = np.random.randn(dim_w, R)
            elif attempt % 4 == 1:
                U = np.random.choice([-1, 0, 1], size=(dim_u, R)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(dim_v, R)).astype(float)
                W = np.random.choice([-1, 0, 1], size=(dim_w, R)).astype(float)
            elif attempt % 4 == 2:
                U = np.random.randn(dim_u, R) * 0.5
                V = np.random.randn(dim_v, R) * 0.5
                W = np.random.randn(dim_w, R) * 0.5
            else:
                # Start from schoolbook and truncate
                U_sb = np.zeros((dim_u, n * m * p))
                V_sb = np.zeros((dim_v, n * m * p))
                W_sb = np.zeros((dim_w, n * m * p))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_sb[i * m + j, col] = 1
                            V_sb[j * p + k, col] = 1
                            W_sb[k * n + i, col] = 1
                            col += 1
                # Keep first R columns, add noise
                U = U_sb[:, :R] + np.random.randn(dim_u, R) * 0.1
                V = V_sb[:, :R] + np.random.randn(dim_v, R) * 0.1
                W = W_sb[:, :R] + np.random.randn(dim_w, R) * 0.1

            result = try_polish_and_verify(U, V, W, R)
            if result is not None:
                rank_found, sol = result
                if rank_found < best_rank:
                    best_rank = rank_found
                    best_solution = sol
                found = True
                break

        if found:
            break

    # ============================================================
    # Strategy 3: Try rank 25-29 with more attempts
    # ============================================================

    for R in range(29, 24, -1):
        if time.time() - start_time > time_limit:
            break
        if R >= best_rank:
            continue

        found = False
        for attempt in range(30):
            if time.time() - start_time > time_limit:
                break

            U = np.random.randn(dim_u, R) * 0.3
            V = np.random.randn(dim_v, R) * 0.3
            W = np.random.randn(dim_w, R) * 0.3

            result = try_polish_and_verify(U, V, W, R)
            if result is not None:
                rank_found, sol = result
                if rank_found < best_rank:
                    best_rank = rank_found
                    best_solution = sol
                found = True
                break

        if found:
            break

    return best_solution
# EVOLVE-BLOCK-END
