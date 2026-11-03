import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill limit

    def check_solution(U, V, W, R):
        """Check if the decomposition is exact. Returns max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def check_solution_fast(U, V, W, R):
        """Fast check using vectorized operations. Returns max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                prod = np.zeros(R)
                for k in range(p):
                    prod += V[j*p + k, :] * W[k*n + i, :]
                val = np.dot(U[i*m + j, :], prod)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
        return max_err

    def check_solution_all(U, V, W, R):
        """Check ALL (i,j,k) terms. Returns max absolute error over all required terms."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term should be 1 if (i,j,k) corresponds to a valid multiplication term
                    # In matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
                    # The tensor has T[i,j,k] = 1 for all i,j,k (it's the identity pattern)
                    # Actually for 2x3x5: T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,5)
                    # The decomposition must satisfy: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                    prod = np.zeros(R)
                    for kk in range(p):
                        pass
                    # For the (i,j,k) term specifically:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_strassen_rank28():
        """Build explicit rank-28 solution using two Strassen 2x2x2 blocks."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        c = 0

        def add_strassen_block(j_off, k_off):
            nonlocal c
            # m1 = (a00+a11)*(b00+b11)
            U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1

            # m2 = (a10+a11)*b00
            U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1
            W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
            c += 1

            # m3 = a00*(b01-b11)
            U[0*m + j_off, c] = 1
            V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
            W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1

            # m4 = a11*(b10-b00)
            U[1*m + (j_off+1), c] = 1
            V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
            W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
            c += 1

            # m5 = (a00+a01)*b11
            U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
            V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
            c += 1

            # m6 = (a10-a00)*(b00+b01)
            U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
            V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
            W[(k_off+1)*n + 1, c] = 1
            c += 1

            # m7 = (a01-a11)*(b10+b11)
            U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
            V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1
            c += 1

        # Block 1: j in {0,1}, k in {0,1}
        add_strassen_block(0, 0)

        # Block 2: j in {0,1}, k in {2,3}
        add_strassen_block(0, 2)

        # Remaining 14 schoolbook terms (j=2, all k; plus j=0,1 with k=4)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j in (0, 1) and k in (0, 1, 2, 3):
                        continue
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1

        assert c == rank, f"Expected rank {rank}, got {c}"
        return U, V, W, rank

    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W

    def als_optimize(U, V, W, R, max_iters=200, tol=1e-12):
        """Run ALS optimization. Returns (U, V, W, err)."""
        prev_err = float('inf')
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution_fast(U, V, W, R)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
            # Early termination if no improvement over last 5 iterations
            if it > 15 and err > best_err * 1.001 and err > prev_err * 0.999:
                break
            prev_err = err
        return best_U, best_V, best_W, best_err

    def polish(U, V, W, R, max_iters=500, tol=1e-14):
        """High-precision polish pass with tighter convergence tracking."""
        prev_err = float('inf')
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
            # Detect stagnation
            if it > 5 and abs(prev_err - err) < 1e-16 * max(1.0, prev_err):
                break
            prev_err = err
        return U, V, W, err

    def try_rank27_from_perturbation(U28, V28, W28, rank28, R, trial_idx, deadline):
        """Try to find rank-R solution by perturbing rank-28 solution."""
        if time.time() > deadline - 5:
            return None
        
        # Strategy 1: Zero out a random column
        col_to_zero = trial_idx % rank28
        U = U28.copy()
        V = V28.copy()
        W = W28.copy()
        U[:, col_to_zero] = 0
        V[:, col_to_zero] = 0
        W[:, col_to_zero] = 0
        U = np.delete(U, col_to_zero, axis=1)
        V = np.delete(V, col_to_zero, axis=1)
        W = np.delete(W, col_to_zero, axis=1)
        
        # Add small perturbation to break symmetry
        np.random.seed(trial_idx * 104729 + 41)
        scale = 0.1 * np.exp(-trial_idx / 50.0)
        U += np.random.randn(n*m, R) * scale
        V += np.random.randn(m*p, R) * scale
        W += np.random.randn(p*n, R) * scale
        
        U, V, W, err = als_optimize(U, V, W, R, max_iters=150, tol=1e-12)
        if err < 1e-10:
            U, V, W, err = polish(U, V, W, R, max_iters=300, tol=1e-14)
            if err < 1e-8:
                # Final verification with full check
                final_err = check_solution_all(U, V, W, R)
                if final_err < 1e-8:
                    return U, V, W
        return None

    # Step 1: Build guaranteed rank-28 solution
    U28, V28, W28, rank28 = build_strassen_rank28()
    err28 = check_solution(U28, V28, W28, rank28)

    # Verify rank-28 is correct
    if err28 > 1e-10:
        # Fallback to schoolbook rank 30
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Step 2: Try to find rank-27 solution
    best_rank = rank28
    best_U, best_V, best_W = U28, V28, W28

    if time.time() < deadline - 10:
        R = 27

        # Strategy A: Perturb rank-28 by zeroing each column with diverse perturbations
        for trial_idx in range(30):
            if time.time() > deadline - 5:
                break
            result = try_rank27_from_perturbation(U28, V28, W28, rank28, R, trial_idx, deadline)
            if result is not None:
                best_rank = R
                best_U, best_V, best_W = result
                break

        # Strategy B: Random restarts with diverse seeds and scaled initialization
        if best_rank == rank28 and time.time() < deadline - 5:
            n_trials = 2000
            for trial in range(n_trials):
                if time.time() > deadline - 3:
                    break
                np.random.seed(trial * 7919 + 13)
                
                # Use scaled random initialization to avoid poor local minima
                scale = 1.0 + 0.5 * np.sin(trial * 0.1)
                U = np.random.randn(n*m, R) * scale
                V = np.random.randn(m*p, R) * scale
                W = np.random.randn(p*n, R) * scale
                
                # Normalize to prevent overflow
                U_norm = np.linalg.norm(U)
                V_norm = np.linalg.norm(V)
                W_norm = np.linalg.norm(W)
                if U_norm > 0: U /= U_norm
                if V_norm > 0: V /= V_norm
                if W_norm > 0: W /= W_norm
                
                U, V, W, err = als_optimize(U, V, W, R, max_iters=80, tol=1e-12)
                if err < 1e-10:
                    U, V, W, err = polish(U, V, W, R, max_iters=300, tol=1e-14)
                    if err < 1e-8:
                        # Final verification with full check
                        final_err = check_solution_all(U, V, W, R)
                        if final_err < 1e-8:
                            best_rank = R
                            best_U, best_V, best_W = U, V, W
                            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
