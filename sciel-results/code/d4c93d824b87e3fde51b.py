"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 145  # 145s wall-clock limit (5s safety margin before 150s kill)

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (12, rank)
    # W index convention: W[k*n + i, r]

    def _build_strassen_rank54():
        """Build the Strassen-based rank-54 decomposition."""
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # 7 Strassen multiplications for 2x2 * 2x2 -> 2x2
            # U[i*m+j], V[j*p+k], W[k*n+i]

            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base+1) * m + (j_base+1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base+1) * p + (k_base+1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base+1) * n + (i_base+1), col] += 1
            col += 1

            # M1: (A10+A11)B00
            U[(i_base+1) * m + j_base, col] += 1
            U[(i_base+1) * m + (j_base+1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base+1), col] += 1
            W[(k_base+1) * n + (i_base+1), col] -= 1
            col += 1

            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base+1), col] += 1
            V[(j_base+1) * p + (k_base+1), col] -= 1
            W[(k_base+1) * n + i_base, col] += 1
            W[(k_base+1) * n + (i_base+1), col] += 1
            col += 1

            # M3: A11(B10-B00)
            U[(i_base+1) * m + (j_base+1), col] += 1
            V[(j_base+1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base+1), col] += 1
            col += 1

            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base+1), col] += 1
            V[(j_base+1) * p + (k_base+1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[k_base+1 * n + i_base, col] += 1
            col += 1

            # M5: (A10-A00)(B00+B01)
            U[(i_base+1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base+1), col] += 1
            W[(k_base+1) * n + (i_base+1), col] += 1
            col += 1

            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base+1), col] += 1
            U[(i_base+1) * m + (j_base+1), col] -= 1
            V[(j_base+1) * p + k_base, col] += 1
            V[(j_base+1) * p + (k_base+1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            # 2x1 * 1x2 -> 2x2, requires 4 scalar multiplications
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base+1), col] += 1
            W[(k_base+1) * n + i_base, col] += 1
            col += 1

            U[(i_base+1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base+1), col] += 1
            col += 1

            U[(i_base+1) * m + j_base, col] += 1
            V[j_base * p + (k_base+1), col] += 1
            W[(k_base+1) * n + (i_base+1), col] += 1
            col += 1

        # Apply Strassen blocks for j-pairs (0,1) and (2,3), k-pairs (0,1), (2,3), (4,5)
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                add_strassen_block(0, j_base, k_base)

        # Apply naive blocks for j=4, k-pairs (0,1), (2,3), (4,5)
        for k_base in [0, 2, 4]:
            add_naive_2112_block(0, 4, k_base)

        # Trim to actual rank used
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        return U, V, W, col

    def _verify(U, V, W, n, m, p, seed=42):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        r = U.shape[1]
        rng = np.random.RandomState(seed)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r_idx in range(r):
            U_r = U[:, r_idx].reshape(n, m)
            V_r = V[:, r_idx].reshape(m, p)
            W_r = W[:, r_idx].reshape(p, n)
            UA = U_r * A_test
            VB = V_r * B_test
            temp = UA @ VB
            C_computed += temp * W_r.T

        return np.max(np.abs(C_computed - C_expected))

    def _verify_exact(U, V, W, n, m, p):
        """Exact verification: check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def _als_optimize(U_init, V_init, W_init, target_rank, max_iter=100, tol=1e-12):
        """Alternating Least Squares optimization for tensor decomposition.
        
        We want: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
        
        ALS: fix two factors, solve linear system for the third.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        r = target_rank

        for iteration in range(max_iter):
            if _time_remaining() < 10:
                break

            # Update U: for each (i,j), solve M @ U[i*m+j,:] = ones(p)
            # where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        pass

            # Update V: for each (j,k), solve M @ V[j*p+k,:] = ones(n)
            # where M[i, r] = U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    except:
                        pass

            # Update W: for each (k,i), solve M @ W[k*n+i,:] = ones(m)
            # where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        pass

            # Check convergence
            if _time_remaining() < 10:
                break

        return U, V, W

    def _polish(U, V, W, max_iter=50, tol=1e-14):
        """High-precision polish pass with tighter tolerance."""
        r = U.shape[1]
        for iteration in range(max_iter):
            if _time_remaining() < 5:
                break

            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        pass

        return U, V, W

    # Step 1: Build initial Strassen rank-54 solution
    U0, V0, W0, rank0 = _build_strassen_rank54()

    # Verify initial solution
    err0 = _verify(U0, V0, W0, n, m, p)
    if err0 > 1e-8:
        # Fallback to schoolbook if Strassen is broken
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_school[i*m+j, idx] = 1
                    V_school[j*p+k, idx] = 1
                    W_school[k*n+i, idx] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Step 2: Try to reduce rank via ALS optimization
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    best_err = err0

    # Try target ranks from 50 down to 40
    for target_rank in range(50, 39, -1):
        if _time_remaining() < 20:
            break

        # Multiple random restarts for each target rank
        n_restarts = 3
        for restart in range(n_restarts):
            if _time_remaining() < 15:
                break

            if restart == 0:
                # Initialize by truncating the known solution
                U_init = U0[:, :target_rank].copy()
                V_init = V0[:, :target_rank].copy()
                W_init = W0[:, :target_rank].copy()
            else:
                # Random initialization with small noise
                rng = np.random.RandomState(42 + restart)
                U_init = rng.randn(n*m, target_rank) * 0.5
                V_init = rng.randn(m*p, target_rank) * 0.5
                W_init = rng.randn(p*n, target_rank) * 0.5

            # ALS optimization
            U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_init, target_rank, max_iter=80)

            # Verify
            err = _verify(U_opt, V_opt, W_opt, n, m, p, seed=42 + restart)

            if err < 1e-10:
                # Solution is valid!
                # Polish for higher precision
                U_pol, V_pol, W_pol = _polish(U_opt, V_opt, W_opt, max_iter=30)
                err_pol = _verify(U_pol, V_pol, W_pol, n, m, p, seed=42 + restart)
                if err_pol < 1e-10:
                    U_opt, V_opt, W_opt = U_pol, V_pol, W_pol
                    err = err_pol

                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_err = err
                    break  # Found a valid solution at this rank, move to next lower rank

    # Step 3: Try even lower ranks if time permits
    if _time_remaining() > 30:
        for target_rank in range(39, 30, -1):
            if _time_remaining() < 15:
                break

            n_restarts = 2
            for restart in range(n_restarts):
                if _time_remaining() < 10:
                    break

                rng = np.random.RandomState(100 + restart)
                U_init = rng.randn(n*m, target_rank) * 0.5
                V_init = rng.randn(m*p, target_rank) * 0.5
                W_init = rng.randn(p*n, target_rank) * 0.5

                U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_opt, target_rank, max_iter=60)
                err = _verify(U_opt, V_opt, W_opt, n, m, p, seed=100 + restart)

                if err < 1e-10:
                    U_pol, V_pol, W_pol = _polish(U_opt, V_opt, W_opt, max_iter=20)
                    err_pol = _verify(U_pol, V_pol, W_pol, n, m, p, seed=100 + restart)
                    if err_pol < 1e-10:
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_pol, V_pol, W_pol
                            best_err = err_pol

    # Final verification of best solution
    final_err = _verify(best_U, best_V, best_W, n, m, p, seed=12345)
    if final_err > 1e-8:
        # Fallback to the known-good Strassen solution
        final_err = _verify(U0, V0, W0, n, m, p, seed=12345)
        if final_err > 1e-8:
            # Schoolbook fallback
            rank_school = n * m * p
            U_school = np.zeros((n * m, rank_school))
            V_school = np.zeros((m * p, rank_school))
            W_school = np.zeros((p * n, rank_school))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U_school[i*m+j, idx] = 1
                        V_school[j*p+k, idx] = 1
                        W_school[k*n+i, idx] = 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        best_U, best_V, best_W = U0, V0, W0

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
